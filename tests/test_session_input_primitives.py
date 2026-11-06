"""The primitives read only the names core writes, unless a deployment opts in.

Records written before this module existed are a deployment's own business:
``register_legacy_names`` adds the keys such a deployment wrote, and core names
no concrete old key itself.
"""
import pytest

import session_input_primitives as primitives


@pytest.fixture(autouse=True)
def _no_registered_names():
    primitives.clear_legacy_names()
    yield
    primitives.clear_legacy_names()


def _record(text='one'):
    return {'parts': [{'at': 1.0, 'text': text}], 'deadline': 2.0}


def test_core_reads_only_the_names_it_writes():
    assert primitives.input_payload_fields() == frozenset({primitives.META})
    assert primitives.batch_record({primitives.META: _record()})[0] == primitives.META
    assert primitives.batch_record({'old_batch_v0': _record()}) == (None, None)


def test_registered_names_are_read_after_the_core_key():
    primitives.register_legacy_names(batch_keys=['old_batch_v0'], alias_prefixes=['old.alias.v0.'])

    assert primitives.input_payload_fields() == frozenset({primitives.META, 'old_batch_v0'})
    assert primitives.batch_record({'old_batch_v0': _record('old')}) == ('old_batch_v0', _record('old'))
    assert primitives.legacy_alias_prefixes() == ('old.alias.v0.',)
    # The core key still wins, so a live batch is never re-read through an old name.
    both = {'old_batch_v0': _record('old'), primitives.META: _record('new')}
    assert primitives.batch_record(both)[0] == primitives.META


def test_registration_is_idempotent_and_clears_to_the_default():
    primitives.register_legacy_names(batch_keys=['old_batch_v0'])
    primitives.register_legacy_names(batch_keys=['old_batch_v0'])

    assert primitives.legacy_batch_keys() == ('old_batch_v0',)

    primitives.clear_legacy_names()

    assert primitives.legacy_batch_keys() == ()
    assert primitives.batch_record({'old_batch_v0': _record()}) == (None, None)
