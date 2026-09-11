"""Interactive CLI adapter for the shared Collective Wisdom command flow."""


class CLIWisdomMixin:
    def _handle_wisdom_command(self, cmd_original: str) -> None:
        from gateway.wisdom_command import (
            WisdomCommandContext,
            WisdomCommandController,
            command_error_text,
            render_local_view,
        )
        from hermes_constants import get_hermes_home
        from hermes_wisdom.entitlement import require_entitlement
        from hermes_wisdom.service import WisdomService

        parts = cmd_original.split(None, 1)
        invoked_as = parts[0].lstrip("/").lower() if parts else "wisdom"
        raw_args = parts[1].strip() if len(parts) > 1 else ""
        if invoked_as == "collective-wisdom-install":
            raw_args = f"install {raw_args}".strip()

        try:
            require_entitlement()
            service = WisdomService()
            context = WisdomCommandContext(
                user_id="local-user",
                chat_id=f"local:{getattr(self, 'session_id', '')}",
                profile=str(get_hermes_home()),
                organization_id=service.store.active_org_id(),
                is_group=False,
            )
            view = WisdomCommandController().execute(raw_args, service, context)
            print(render_local_view(view, context))
        except Exception as exc:
            print(f"Collective Wisdom could not continue: {command_error_text(exc)}")
