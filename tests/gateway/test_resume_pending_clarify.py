"""``session.resume`` republishes a waiting clarify in the legacy ``pending_clarify`` map as well.

`gateway/session_controls.resume` adds `reply['pending_clarify']` so a client built before the
canonical `prompts` array (Hermes Mobile reads `pending_clarify` and takes the id from
`clarify_id`/`request_id`) can restore a card for a question asked while it was detached — without
it the turn waited out `clarify_timeout` with nothing on screen to answer. The field has to be
declared on the resume contract, because `Result` forbids extra keys: declaring it on
`LiveSessionSnapshot` alone left every such resume logging a wire-contract violation
(`Extra inputs are not permitted [pending_clarify]`) while still delivering the card.
"""
import pytest
from pydantic import ValidationError

from gateway.session_pending_clarify import pending_clarify_payload
from tui_gateway.contracts.canonical_sessions import CanonicalSnapshot
from tui_gateway.contracts.registry import CANONICAL_METHODS
from tui_gateway.contracts.sessions import LiveSessionSnapshot, PendingClarify

SID = 'local-session'


def _prompt(prompt_id='ef27355965', generation=3, **extra):
    return {'kind': 'clarify', 'prompt_id': prompt_id, 'execution_generation': generation,
            'question': 'Какой путь?', 'choices': ['кратчайший', 'безопасный'], 'multi_select': False,
            'timeout_s': 1800.0, **extra}


def _extra_forbidden(errors, key):
    return [e for e in errors if e.get('type') == 'extra_forbidden' and e.get('loc', ('',))[0] == key]


def _errors(payload, model=CanonicalSnapshot):
    try:
        model.model_validate(payload)
    except ValidationError as exc:
        return exc.errors()
    return []


# ── the projection ────────────────────────────────────────────────────────────────────────────


def test_a_pending_clarify_is_projected_into_the_legacy_shape():
    payload = pending_clarify_payload([_prompt()], SID)
    assert payload is not None
    assert payload == {'clarify_id': 'ef27355965', 'request_id': 'ef27355965', 'session_id': SID,
                       'question': 'Какой путь?', 'choices': ['кратчайший', 'безопасный'],
                       'multi_select': False, 'execution_generation': 3, 'timeout_s': 1800.0}


def test_the_card_id_is_the_prompt_id_the_answer_must_be_fenced_with():
    # The client echoes ``clarify_id`` back; the answer verb resolves it to the same prompt, so the
    # restored card and the live question name one prompt and one generation.
    payload = pending_clarify_payload([_prompt('p-7', 11)], SID)
    assert payload is not None
    assert payload['clarify_id'] == payload['request_id'] == 'p-7'
    assert payload['execution_generation'] == 11


def test_nothing_waiting_projects_nothing():
    assert pending_clarify_payload([], SID) is None
    assert pending_clarify_payload(None, SID) is None


def test_only_a_clarify_is_projected_never_an_approval():
    # Approvals restore from their own ``pending_approval`` key; folding one in here would put a
    # command prompt on screen as a question, answered by the wrong verb.
    approval = {'kind': 'approval', 'prompt_id': 'a-1', 'execution_generation': 2,
                'question': 'rm -rf build?', 'choices': ['allow', 'deny']}
    assert pending_clarify_payload([approval], SID) is None


def test_the_first_waiting_question_wins_and_malformed_entries_are_skipped():
    prompts = [{'kind': 'clarify', 'execution_generation': 1},          # no prompt_id
               'not-a-prompt',
               _prompt('second', 4)]
    payload = pending_clarify_payload(prompts, SID)
    assert payload is not None
    assert payload['clarify_id'] == 'second' and payload['execution_generation'] == 4


def test_a_prompt_without_choices_still_projects_a_question():
    payload = pending_clarify_payload([_prompt(choices=None, multi_select=None, timeout_s=None)], SID)
    assert payload is not None
    assert payload['choices'] == [] and payload['multi_select'] is False
    assert 'timeout_s' not in payload


# ── the contract ─────────────────────────────────────────────────────────────────────────────


def test_the_resume_wire_contract_is_the_canonical_snapshot():
    assert CANONICAL_METHODS['session.resume'].result is CanonicalSnapshot


def test_the_resume_contract_carries_the_legacy_map():
    payload = pending_clarify_payload([_prompt()], SID)
    assert payload is not None
    snapshot = LiveSessionSnapshot.model_validate({'session_id': SID, 'message_count': 0,
                                                   'messages': [], 'info': {}, 'pending_clarify': payload})
    assert snapshot.pending_clarify is not None
    assert snapshot.pending_clarify.clarify_id == 'ef27355965'
    assert snapshot.pending_clarify.choices == ['кратчайший', 'безопасный']


def test_a_resume_carrying_the_legacy_card_is_not_an_extra_key():
    card = pending_clarify_payload([_prompt('a642e5065f', 1)], 'local-s')
    assert card is not None
    assert _extra_forbidden(_errors({'session_id': 'local-s', 'pending_clarify': card}),
                            'pending_clarify') == []


def test_any_other_unknown_key_is_still_refused():
    # The contract stays closed: only the declared key is accepted.
    assert _extra_forbidden(_errors({'session_id': 'local-s', 'pending_clarify_typo': {}}),
                            'pending_clarify_typo')


def test_a_resume_with_no_waiting_question_stays_silent():
    assert _extra_forbidden(_errors({'session_id': 'local-s'}), 'pending_clarify') == []
    snapshot = LiveSessionSnapshot.model_validate({'session_id': SID, 'message_count': 0,
                                                   'messages': [], 'info': {}})
    assert snapshot.pending_clarify is None


@pytest.mark.parametrize('key', ['clarify_id', 'request_id', 'question', 'choices', 'multi_select',
                                 'execution_generation', 'timeout_s'])
def test_the_card_keeps_every_field_the_client_reads(key):
    card = {'clarify_id': 'p1', 'request_id': 'p1', 'session_id': 'local-s', 'question': 'q?',
            'choices': ['a', 'b'], 'multi_select': False, 'execution_generation': 3, 'timeout_s': 1800}
    model = PendingClarify.model_validate(card)
    assert getattr(model, key) is not None
