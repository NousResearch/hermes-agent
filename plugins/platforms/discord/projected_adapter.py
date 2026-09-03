"""Discord adapter entry point with shared projection settlement.

The legacy adapter keeps the accepted interaction callbacks and command-tree
construction.  This bounded subclass changes only the registration factory and
the post-sync proof: desired native commands are fingerprinted through the same
projection contract as relay commands, then Discord is read back and compared
before sync is reported as settled.
"""

from __future__ import annotations

from typing import Any

from gateway.discord_command_projection import (
    DiscordCommandProjection,
    DiscordProjectionMismatch,
    project_discord_commands,
    verify_discord_projection_readback,
)

from . import adapter as _adapter


class ProjectedDiscordAdapter(_adapter.DiscordAdapter):
    """DiscordAdapter with deterministic projection and exact remote read-back."""

    _READBACK_ATTEMPTS = 3
    _SAME_FINGERPRINT_SKIP = "same slash-command fingerprint already synced"

    _discord_command_projection_revision: str | None = None
    _discord_command_projection_verified_revision: str | None = None

    def _desired_command_projection(self) -> DiscordCommandProjection:
        if not self._client:
            return project_discord_commands(())
        tree = self._client.tree
        return project_discord_commands(
            command.to_dict(tree)
            for command in tree.get_commands()
        )

    def _remote_command_payload(self, command: Any) -> dict[str, Any]:
        return self._existing_command_to_payload(command)

    def _desired_command_sync_fingerprint(self) -> str:
        """Persist the shared canonical projection revision as sync identity."""
        return self._desired_command_projection().revision

    def _command_sync_skip_reason(
        self, app_id: Any, fingerprint: str
    ) -> str | None:
        """Retain rate-limit backoff, but re-read Discord on every startup.

        A local fingerprint proves only what this process intended last time.
        It cannot prove that Discord still holds that object after an external
        edit or partial remote mutation.  The normal safe reconciler is cheap
        when nothing changed, so the same-fingerprint shortcut is converted
        into an exact remote read-back instead of a completion claim.
        """
        reason = super()._command_sync_skip_reason(app_id, fingerprint)
        if reason == self._SAME_FINGERPRINT_SKIP:
            return None
        return reason

    async def _verify_remote_projection(
        self,
        desired: DiscordCommandProjection,
    ) -> DiscordCommandProjection:
        """Read Discord back with a bounded propagation retry."""
        if not self._client:
            return project_discord_commands(())

        last_mismatch = None
        for attempt in range(self._READBACK_ATTEMPTS):
            remote_commands = await self._client.tree.fetch_commands()
            try:
                return verify_discord_projection_readback(
                    desired,
                    (
                        self._remote_command_payload(command)
                        for command in remote_commands
                    ),
                )
            except DiscordProjectionMismatch as exc:
                last_mismatch = exc
                if attempt + 1 >= self._READBACK_ATTEMPTS:
                    raise
                await self._sleep_between_command_sync_mutations()

        if last_mismatch is None:
            raise RuntimeError("Discord read-back ended without a projection result")
        raise last_mismatch

    async def _safe_sync_slash_commands(self) -> dict[str, int]:
        """Run the existing minimal diff, then prove exact remote settlement."""
        desired = self._desired_command_projection()
        result = await super()._safe_sync_slash_commands()
        if not self._client:
            return result

        observed = await self._verify_remote_projection(desired)
        self._discord_command_projection_revision = desired.revision
        self._discord_command_projection_verified_revision = observed.revision
        return result


def _build_projected_adapter(config):
    return ProjectedDiscordAdapter(config)


class _ProjectedRegistrationContext:
    """Narrow proxy that swaps only Discord's adapter factory."""

    def __init__(self, target: Any) -> None:
        self._target = target

    def register_platform(self, *args: Any, **kwargs: Any):
        forwarded = dict(kwargs)
        if forwarded.get("name") == "discord":
            forwarded["adapter_factory"] = _build_projected_adapter
        return self._target.register_platform(*args, **forwarded)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def register(ctx) -> None:
    """Register Discord without copying the adapter's plugin metadata."""
    _adapter.register(_ProjectedRegistrationContext(ctx))


__all__ = ["ProjectedDiscordAdapter", "register"]
