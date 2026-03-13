from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BACKBONE = "neuphonic/neutts-nano"
DEFAULT_CODEC = "neuphonic/neucodec"
DEFAULT_SAMPLE_RATE = 24000


def app_home() -> Path:
    override = os.getenv("NEUTTS_CLI_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".neutts-cli"


def config_path() -> Path:
    return app_home() / "config.json"


def voices_dir() -> Path:
    return app_home() / "voices"


@dataclass
class AppConfig:
    backbone_repo: str = DEFAULT_BACKBONE
    backbone_device: str = "cpu"
    codec_repo: str = DEFAULT_CODEC
    codec_device: str = "cpu"
    sample_rate: int = DEFAULT_SAMPLE_RATE
    default_voice: str | None = None

    @classmethod
    def load(cls) -> "AppConfig":
        path = config_path()
        if not path.exists():
            return cls()

        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            backbone_repo=data.get("backbone_repo", DEFAULT_BACKBONE),
            backbone_device=data.get("backbone_device", "cpu"),
            codec_repo=data.get("codec_repo", DEFAULT_CODEC),
            codec_device=data.get("codec_device", "cpu"),
            sample_rate=int(data.get("sample_rate", DEFAULT_SAMPLE_RATE)),
            default_voice=data.get("default_voice") or None,
        )

    def save(self) -> Path:
        home = app_home()
        home.mkdir(parents=True, exist_ok=True)
        path = config_path()
        payload = {
            "backbone_repo": self.backbone_repo,
            "backbone_device": self.backbone_device,
            "codec_repo": self.codec_repo,
            "codec_device": self.codec_device,
            "sample_rate": self.sample_rate,
            "default_voice": self.default_voice,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path
