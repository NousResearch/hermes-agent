"""Import deployment-owned credentials into the boot profile's native secret store."""
import os
import shutil
from pathlib import Path

from hermes_cli.config import save_env_value


def seed_service_secrets():
    # Scoped gateway turns intentionally cannot read process-only credentials.
    # Only the boot profile receives this key; other profiles provision their own.
    key = 'HINDSIGHT_API_KEY'
    if key in os.environ:
        save_env_value(key, os.environ[key])


def seed_whisper_model():
    """Copy bundled weights into the native cache without replacing user state."""
    from huggingface_hub.constants import HF_HUB_CACHE
    source = Path('/opt/whisper-cache/models--Systran--faster-whisper-base')
    destination = Path(HF_HUB_CACHE) / source.name
    if source.is_dir() and not destination.exists():
        shutil.copytree(source, destination, symlinks=True)


if __name__ == '__main__':
    seed_service_secrets()
    seed_whisper_model()
