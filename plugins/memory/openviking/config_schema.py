"""OpenViking's declarative Desktop configuration surface."""

from plugins.memory.config_schema import (
    KIND_SECRET,
    KIND_SEGMENTED,
    KIND_SELECT,
    ProviderConfigAction,
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldCondition,
    ProviderFieldOption,
)


_SERVICE = ProviderFieldCondition("setup_type", values=("service",))
_PROFILE = ProviderFieldCondition("setup_type", values=("profile",))
_CUSTOM = ProviderFieldCondition("setup_type", values=("custom",))
_MANUAL = ProviderFieldCondition("setup_type", values=("service", "custom"))
_CUSTOM_WITH_KEY = ProviderFieldCondition("credential", values=("user", "root"))
_CUSTOM_ROOT = ProviderFieldCondition("credential", values=("root",))
_LOCAL_URL = ProviderFieldCondition(
    "url",
    pattern=r"^http://(?:127\.0\.0\.1|localhost|\[::1\])(?::\d+)?(?:/|$)",
)


CONFIG_SCHEMA = ProviderConfigSchema(
    name="openviking",
    label="OpenViking",
    description="Configure Hermes memory with OpenViking profiles, service, or a self-hosted server.",
    submit_action="save",
    submit_label="Save setup",
    status_action="health",
    actions=(
        ProviderConfigAction(
            name="start-local",
            label="Start local server",
            description="Start an installed and configured openviking-server process.",
            after_field="url",
            payload_fields=("url",),
            visible_when=(_CUSTOM, _LOCAL_URL),
            refresh_after=True,
        ),
    ),
    fields=(
        ProviderField(
            key="setup_type",
            label="Setup Type",
            kind=KIND_SEGMENTED,
            default="service",
            description="Choose a managed service, an existing OpenViking CLI profile, or another server.",
            required=True,
            options=(
                ProviderFieldOption("service", "OpenViking Service"),
                ProviderFieldOption("profile", "Existing Profiles"),
                ProviderFieldOption("custom", "Custom Server"),
            ),
        ),
        ProviderField(
            key="profile_path",
            label="OpenViking profile",
            kind=KIND_SELECT,
            description="Link this Hermes profile to an OpenViking CLI profile.",
            placeholder="No OpenViking profiles found",
            search_placeholder="Search profiles...",
            required=True,
            dynamic_options=True,
            searchable=True,
            visible_when=(_PROFILE,),
        ),
        ProviderField(
            key="profile_name",
            label="Profile Name",
            description="Saved as an OpenViking CLI profile and linked to this Hermes profile.",
            default="openviking",
            required=True,
            visible_when=(_MANUAL,),
        ),
        ProviderField(
            key="url",
            label="OpenViking URL",
            description="Local or remote OpenViking server.",
            default="http://127.0.0.1:1933",
            required=True,
            visible_when=(_CUSTOM,),
        ),
        ProviderField(
            key="credential",
            label="Credential",
            kind=KIND_SELECT,
            default="user",
            description="Choose how Hermes authenticates with this OpenViking server.",
            visible_when=(_CUSTOM,),
            options=(
                ProviderFieldOption(
                    "none",
                    "No API key",
                    "For an explicitly unauthenticated local server.",
                ),
                ProviderFieldOption(
                    "user", "User API key", "Authenticate as an OpenViking user."
                ),
                ProviderFieldOption(
                    "root",
                    "Root API key",
                    "Authenticate as root for an account and user.",
                ),
            ),
        ),
        ProviderField(
            key="api_key_service",
            label="OpenViking API key",
            kind=KIND_SECRET,
            description="Stored only in the OpenViking CLI profile.",
            required=True,
            help_url="https://console.volcengine.com/vikingdb/openviking/region:openviking+cn-beijing",
            help_label="Get OpenViking API key",
            visible_when=(_SERVICE,),
        ),
        ProviderField(
            key="api_key",
            label="OpenViking API key",
            kind=KIND_SECRET,
            description="Stored only in the OpenViking CLI profile.",
            required=True,
            visible_when=(_CUSTOM, _CUSTOM_WITH_KEY),
        ),
        ProviderField(
            key="account",
            label="Account",
            description="Required when authenticating with a root API key.",
            required=True,
            visible_when=(_CUSTOM, _CUSTOM_ROOT),
        ),
        ProviderField(
            key="user",
            label="User",
            description="Required when authenticating with a root API key.",
            required=True,
            visible_when=(_CUSTOM, _CUSTOM_ROOT),
        ),
        ProviderField(
            key="actor_peer_id",
            label="Agent ID",
            default="",
            description=(
                "Optional peer ID for separate assistant context. "
                "Leave blank to use user memory."
            ),
            visible_when=(_MANUAL,),
        ),
    ),
)
