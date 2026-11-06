"""Opt-in native-card factory for the bundled Feishu platform registration."""

from utils import is_truthy_value

from .state import RoutingHub


def register_card_factory(ctx, base):
    hub = RoutingHub()
    for name, callback in (
        ("pre_tool_call", hub.pre_tool),
        ("post_tool_call", hub.post_tool),
        ("pre_approval_request", hub.pre_approval),
        ("pre_api_request", hub.pre_api),
        ("pre_auxiliary_call", hub.pre_auxiliary),
        ("post_auxiliary_call", hub.post_auxiliary),
    ):
        ctx.register_hook(name, callback)

    def factory(config):
        if not is_truthy_value((config.extra or {}).get("native_cards", False)):
            return base(config)
        from .adapter import build_adapter_class

        return build_adapter_class(base, hub)(config)

    return factory
