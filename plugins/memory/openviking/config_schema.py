"""OpenViking's Desktop configuration surface declaration."""

from plugins.memory.config_schema import ProviderConfigSchema


CONFIG_SCHEMA = ProviderConfigSchema(
    name="openviking",
    label="OpenViking",
    custom_surface="openviking",
)
