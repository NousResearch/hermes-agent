"""Transient per-admission tool environment; deliberately excluded from durable payloads."""
from agent.session_client_env import accepted_turn_env, configured_turn_names
from hermes_state_runtime import RuntimeStoreError


def admit_turn_env(authority, value):
    if value is None:
        return {}
    from gateway.session_policy import profile_config
    try:
        return accepted_turn_env(value, configured_turn_names(profile_config(authority)))
    except ValueError:
        # Never echo a value, including in chained exceptions at an RPC/log boundary.
        raise RuntimeStoreError('invalid_params') from None


def remember_turn_env(authority, row, values, *, new_admission):
    # First admission wins, even if it sent nothing. A resend may change this ephemeral
    # map without changing the durable digest, but may never replace a running turn's
    # values or attach new ones to recovered work after an owner restart.
    if new_admission and row['status'] == 'queued' and values:
        authority.turn_envs[row['admission_id']] = values
