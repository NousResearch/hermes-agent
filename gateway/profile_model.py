"""Helper to read a profile's configured model."""
import os
from pathlib import Path
from typing import Optional

try:
    from hermes_cli.config import load_config_readonly as _load_config_readonly
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
except ImportError:
    _load_config_readonly = None
    set_hermes_home_override = None
    reset_hermes_home_override = None


def _resolve_profile_dir(profile_name: str) -> Path:
    """
    Resolve the profile directory, checking both standard locations.
    
    Standard: ~/.hermes/profiles/<name> or ~/.hermes (for default)
    Windows AppData: %LOCALAPPDATA%\Hermes\profiles\<name> or %LOCALAPPDATA%\Hermes
    """
    # Check standard location first
    if profile_name == "default":
        std_dir = Path.home() / ".hermes"
        appdata_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Hermes"
    else:
        std_dir = Path.home() / ".hermes" / "profiles" / profile_name
        appdata_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Hermes" / "profiles" / profile_name
    
    # Prefer the one that exists and has config.yaml
    for d in [std_dir, appdata_dir]:
        if d.exists() and (d / "config.yaml").exists():
            return d
    
    # Fallback to standard
    return std_dir


def get_profile_active_model(profile_name: str) -> Optional[str]:
    """
    Get the active model string for a given profile.
    
    Reads the profile's config.yaml and returns the model in format
    "provider/model" or just "model" depending on configuration.
    Returns None if not found or on error.
    """
    if not _load_config_readonly or not set_hermes_home_override:
        return None
    
    profile_dir = _resolve_profile_dir(profile_name)
    
    token = set_hermes_home_override(str(profile_dir))
    try:
        cfg = _load_config_readonly()
        model_cfg = cfg.get("model", {})
        
        # Check for explicit provider/model structure
        provider = model_cfg.get("provider")
        model = model_cfg.get("default")
        
        if provider and model:
            return f"{provider}/{model}"
        elif model:
            return model
        else:
            return None
    except Exception:
        return None
    finally:
        if reset_hermes_home_override:
            reset_hermes_home_override(token)


def get_profile_model_info(profile_name: str) -> str:
    """
    Get a formatted model info string for display.
    
    Returns string like "nous/tencent/hy3:free" or "unknown".
    """
    model = get_profile_active_model(profile_name)
    return model if model else "unknown"