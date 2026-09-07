FROM nousresearch/hermes-agent:v2026.8.31@sha256:64923faeae267792bf9bf87fe3b4c4869e35004e360c7df01730ad801b74d524

# The bundled Teams 2.0.13.4 pins MSAL and cryptography incompatibly.
# Keep the base image and all other Hermes code; update only the Teams SDK pin.
RUN sed -i 's/microsoft-teams-apps==2\.0\.13\.4/microsoft-teams-apps==2.0.16/g' \
        /opt/hermes/tools/lazy_deps.py /opt/hermes/pyproject.toml \
    && uv pip install --python /opt/hermes/.venv/bin/python \
        'microsoft-teams-apps==2.0.16' 'microsoft-teams-api==2.0.16' \
        'microsoft-teams-common==2.0.16' 'microsoft-teams-cards==2.0.16' \
        'dependency-injector==4.49.1' 'pydantic-settings==2.15.0' \
        'aiohttp==3.14.3'

# Verify the real adapter imports and lazy installer accepts the installed SDK.
RUN HERMES_HOME=/tmp/hermes-teams-build-check /opt/hermes/.venv/bin/python -c \
    "from tools.lazy_deps import feature_missing; from plugins.platforms.teams.adapter import check_teams_requirements; assert not feature_missing('platform.teams'); assert check_teams_requirements()"
