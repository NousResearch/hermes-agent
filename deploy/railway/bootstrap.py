"""Import deployment-owned credentials into the boot profile's native secret store."""
import os

from hermes_cli.config import save_env_value


def seed_service_secrets():
    # Scoped gateway turns intentionally cannot read process-only credentials.
    # Only the boot profile receives this key; other profiles provision their own.
    key = 'HINDSIGHT_API_KEY'
    if key in os.environ:
        save_env_value(key, os.environ[key])


if __name__ == '__main__':
    seed_service_secrets()
