from . import Copy

COPY = Copy(
    legacy='hermes_state_gateway',
    doc='Gateway session listing.',
    legacy_class='SessionGatewayMixin',
    members=(
        'list_gateway_sessions',
    ),
)
