"""The pending clarify as the legacy ``pending_clarify`` map a client restores a card from.

``session.resume`` reports a blocked turn's question only in the canonical ``prompts`` array
(``{kind, prompt_id, execution_generation, question, choices, multi_select}``, published by
``gateway/session_pending_controls.py::register_clarify``). Clients built against the older shape
never look there: Hermes Mobile (``com.m57.hermescontrol``) restores a card from a
``pending_clarify`` map (top level and inside ``info``) or from a ``method="clarify"`` entry in
``open_requests``, and its parser takes the id from ``clarify_id``/``request_id`` — fields the
canonical prompt does not carry.

A question asked while such a client was detached (backgrounded, socket dropped, app restarted)
was therefore a card that never appeared: the session showed a running turn with nothing to answer,
and the turn waited out ``clarify_timeout``. This module projects the live pending prompt into that
legacy shape; the answer still travels through the canonical verb, so the fence stays the pending
prompt's own ``execution_generation``.
"""


def pending_clarify_payload(prompts, session_id):
    """The first waiting clarify in the legacy shape, or ``None`` when nothing is waiting.

    Computed from the live controls snapshot, so a resolved prompt disappears on the next resume.
    Approvals are never folded in here: they restore from their own ``pending_approval`` key, and a
    command prompt rendered as a question would be answered by the wrong verb.
    """
    for prompt in prompts or ():
        if not isinstance(prompt, dict) or prompt.get('kind') != 'clarify':
            continue
        prompt_id = prompt.get('prompt_id')
        if not isinstance(prompt_id, str) or not prompt_id:
            continue
        payload = {'clarify_id': prompt_id, 'request_id': prompt_id, 'session_id': session_id,
                   'question': prompt.get('question') or '',
                   'choices': [c for c in prompt.get('choices') or () if isinstance(c, str)],
                   'multi_select': bool(prompt.get('multi_select')),
                   'execution_generation': prompt.get('execution_generation')}
        if prompt.get('timeout_s') is not None:
            payload['timeout_s'] = prompt['timeout_s']
        return payload
    return None
