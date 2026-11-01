"""A renamed profile's gateway-owned records, re-owned by the profile at its new home.

The authority serving a profile is owned by the profile's resolved home path (``gateway/run_runtime.py``),
and the profile's own store keeps four kinds of record that name that owner:

* LOCAL creation receipts (``gateway.local_policy.v1:<sid>``): ``profile_id``, the ``agent:<name>:`` route
  and entry, the origin profile, and the frozen credential reference (``profile-config-v1:{home, profile}``),
  whose scope every turn launch checks (``gateway/session_policy_credentials._check_scope``);
* their legacy bindings (``gateway.local_legacy.v1:<sid>``), which must equal the receipt;
* API bindings (``gateway.api.binding.v1.<sid>``): ``profile_id``, the route and its entry;
* imported history's control bindings (``gateway.history_control.v1.<sid>``): ``[profile_id, subject]``.

``rename_profile`` moves the directory, so after a rename every one of them names a home that no longer
exists, and the renamed profile's authority refuses them all (``profile_mismatch``). The rename's identity
migration (``SessionDB.rekey_profile_state``) re-owns them in the transaction that moves the session rows and
routing entries they must agree with. Only a record naming exactly the old home moves: one from another
machine or another home is not this profile's to adopt, and keeps refusing.

A receipt's session id is derived from the profile it was created under (``hermes_state_keys.local_identity``),
so a re-owned receipt keeps that profile as ``created_profile_id`` (``hermes_state_local.creation_identity``).
The id, the transcript and every client's reference to the chat stay what they were.
"""
import json
from pathlib import Path

from hermes_state_runtime import _json


def renamed_owner_ids(db_path, old_name, new_name):
    """``(old, new)`` owner ids of profile *new_name*, formerly *old_name*, when *db_path* is its own store:
    its resolved home, as the serving authority spells it, before and after the directory moved. None for
    any other store (the launch root's). Resolved, so a symlinked ``profiles/`` still names the profile."""
    home = Path(db_path).resolve().parent
    if home.name != new_name:
        return None
    return str(home.with_name(old_name)), str(home)


class _Move:
    """One rename: the owner ids, ``agent:<name>:`` namespaces and profile names it moves between."""

    def __init__(self, old_id, new_id, old_ns, new_ns, old_name, new_name):
        self.old_id, self.new_id = old_id, new_id
        self.old_ns, self.new_ns = old_ns, new_ns
        self.old_name, self.new_name = old_name, new_name

    def key(self, session_key):
        if isinstance(session_key, str) and session_key.startswith(self.old_ns):
            return self.new_ns + session_key[len(self.old_ns):]
        return session_key

    def entry(self, entry):
        """A stored routing entry, moved as ``rekey_profile_state`` moves the routing index's own."""
        if 'session_key' in entry:
            entry['session_key'] = self.key(entry['session_key'])
        origin = entry.get('origin')
        if isinstance(origin, dict) and origin.get('profile') == self.old_name:
            origin['profile'] = self.new_name

    def owned(self, record):
        """*record* (a receipt or an API binding) re-owned with its route, else None when not the old owner's."""
        if not isinstance(record, dict) or record.get('profile_id') != self.old_id:
            return None
        record['profile_id'] = self.new_id
        if 'route' in record:
            record['route'] = self.key(record['route'])
        if isinstance(record.get('entry'), dict):
            self.entry(record['entry'])
        return record


def _receipt(move, receipt):
    if not isinstance(receipt, dict):
        return None
    created = receipt.get('created_profile_id', receipt.get('profile_id'))
    if move.owned(receipt) is None:
        return None
    if 'legacy_session_id' not in receipt:
        receipt['created_profile_id'] = created
        if created == move.new_id:  # renamed back: its id derives from its owner again
            del receipt['created_profile_id']
    policy = receipt.get('policy')
    ref = policy.get('config_secret_ref') if isinstance(policy, dict) else None
    from gateway.session_policy_credentials import PREFIX
    source = json.loads(ref[len(PREFIX):]) if isinstance(ref, str) and ref.startswith(PREFIX) else None
    if isinstance(source, dict):
        for field in ('home', 'profile'):
            if source.get(field) == move.old_id:
                source[field] = move.new_id
        # The fingerprints and endpoint bindings stay as frozen: recovery still checks each secret
        # against the profile's config.yaml, so a key the user changed since keeps refusing.
        policy['config_secret_ref'] = PREFIX + json.dumps(source, sort_keys=True)
    return _json(receipt)


def _legacy_binding(move, binding):
    if not isinstance(binding, dict) or binding.get('profile_id') != move.old_id:
        return None
    return _json(dict(binding, profile_id=move.new_id))


def _api_binding(move, binding):
    return None if move.owned(binding) is None else _json(binding)


def _history_binding(move, binding):
    if not isinstance(binding, list) or len(binding) != 2 or binding[0] != move.old_id:
        return None
    # ``authorize_history`` compares the stored text to the actor's, so the spelling is its own.
    return json.dumps([move.new_id, binding[1]], separators=(',', ':'))


def _kinds():
    from hermes_state_local import API_BINDING_PREFIX, POLICY_PREFIX
    from hermes_state_local_migration import LEGACY_PREFIX
    from hermes_state_mutation_binding import BINDING_PREFIX
    return {'local_receipts': (POLICY_PREFIX, _receipt),
            'local_legacy_bindings': (LEGACY_PREFIX, _legacy_binding),
            'api_bindings': (API_BINDING_PREFIX, _api_binding),
            'history_bindings': (BINDING_PREFIX, _history_binding)}


def rekey_owner_records(conn, *, old_id, new_id, old_ns, new_ns, old_name, new_name):
    """Re-own, on *conn*'s write transaction, every record of this store naming owner *old_id*: ``{kind: n}``.
    A record that does not parse is left as it is (its own readers already refuse it)."""
    move = _Move(old_id, new_id, old_ns, new_ns, old_name, new_name)
    counts = {}
    for kind, (prefix, rekey) in _kinds().items():
        moved = []
        for key, value in conn.execute('SELECT key, value FROM state_meta WHERE key GLOB ?', (prefix + '*',)).fetchall():
            try:
                rewritten = rekey(move, json.loads(value))
            except (ValueError, TypeError):
                continue
            if rewritten is not None:
                moved.append((rewritten, key))
        conn.executemany('UPDATE state_meta SET value=? WHERE key=?', moved)
        counts[kind] = len(moved)
    return counts
