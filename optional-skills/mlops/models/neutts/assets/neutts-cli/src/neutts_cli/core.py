from __future__ import annotations

import importlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .audio import write_wav
from .config import AppConfig, app_home, voices_dir


KNOWN_MODELS = [
    "neuphonic/neutts-air",
    "neuphonic/neutts-air-q8-gguf",
    "neuphonic/neutts-air-q4-gguf",
    "neuphonic/neutts-nano",
    "neuphonic/neutts-nano-q8-gguf",
    "neuphonic/neutts-nano-q4-gguf",
    "neuphonic/neutts-nano-french",
    "neuphonic/neutts-nano-german",
    "neuphonic/neutts-nano-spanish",
]


@dataclass
class VoiceProfile:
    name: str
    ref_audio: str
    ref_text: str
    language: str = "unknown"


def is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def run_install(
    include_llama: bool, include_onnx: bool, include_all: bool, dry_run: bool
) -> list[str]:
    extras = []
    if include_all:
        extras = ["all"]
    else:
        if include_llama:
            extras.append("llama")
        if include_onnx:
            extras.append("onnx")

    requirement = "neutts"
    if extras:
        requirement = f"neutts[{','.join(extras)}]"

    command = [sys.executable, "-m", "pip", "install", "-U", requirement]
    rendered = " ".join(command)
    if dry_run:
        return [rendered]

    subprocess.run(command, check=True)
    return [rendered]


def platform_notes() -> list[str]:
    system = platform.system()
    if system == "Darwin":
        return [
            "For Apple Silicon GGUF acceleration, install the llama extra with BLAS/Accelerate flags.",
            "See the upstream NeuTTS README for the recommended CMAKE_ARGS invocation.",
        ]
    if system == "Linux":
        return [
            "For GGUF acceleration on Linux, install OpenBLAS and then reinstall the llama extra with matching CMAKE_ARGS.",
        ]
    if system == "Windows":
        return [
            "For GGUF acceleration on Windows, install OpenBLAS first and then install the llama extra from PowerShell with CMAKE_ARGS set.",
        ]
    return []


def doctor_report() -> dict:
    voice_count = (
        len(list(voices_dir().glob("*/voice.json"))) if voices_dir().exists() else 0
    )
    config = AppConfig.load()
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "app_home": str(app_home()),
        "config": asdict(config),
        "neutts_installed": is_module_available("neutts"),
        "numpy_installed": is_module_available("numpy"),
        "onnxruntime_installed": is_module_available("onnxruntime"),
        "llama_cpp_installed": is_module_available("llama_cpp"),
        "ffmpeg_in_path": shutil.which("ffmpeg") is not None,
        "voice_profiles": voice_count,
        "default_voice": config.default_voice,
    }
    return report


def save_voice(
    name: str, ref_audio: str, ref_text: str, language: str = "unknown"
) -> Path:
    source_audio = Path(ref_audio).expanduser().resolve()
    if not source_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {source_audio}")

    destination = voices_dir() / name
    destination.mkdir(parents=True, exist_ok=True)
    audio_target = destination / source_audio.name
    text_target = destination / "reference.txt"
    metadata_target = destination / "voice.json"

    if audio_target.resolve() != source_audio:
        if audio_target.exists():
            audio_target.unlink()
        audio_target.write_bytes(source_audio.read_bytes())
    if text_target.exists():
        text_target.unlink()
    text_target.write_text(ref_text.strip() + "\n", encoding="utf-8")

    profile = VoiceProfile(
        name=name,
        ref_audio=str(audio_target),
        ref_text=ref_text.strip(),
        language=language,
    )
    metadata_target.write_text(
        json.dumps(asdict(profile), indent=2) + "\n", encoding="utf-8"
    )
    return metadata_target


def load_voice(name: str) -> VoiceProfile:
    metadata_path = voices_dir() / name / "voice.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Voice profile not found: {name}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return VoiceProfile(**payload)


def list_voices() -> list[VoiceProfile]:
    if not voices_dir().exists():
        return []

    profiles = []
    for metadata_path in sorted(voices_dir().glob("*/voice.json")):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        profiles.append(VoiceProfile(**payload))
    return profiles


def synthesize(
    text: str,
    out: str,
    voice: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> Path:
    if not text.strip():
        raise ValueError("Input text is required")

    # Fall back to the configured default voice when no voice is specified
    if not voice and not ref_audio:
        config = AppConfig.load()
        if config.default_voice:
            voice = config.default_voice

    if voice:
        profile = load_voice(voice)
        ref_audio = profile.ref_audio
        ref_text = profile.ref_text

    if not ref_audio or not ref_text:
        raise ValueError("Provide either --voice or both --ref-audio and --ref-text")

    if not is_module_available("neutts"):
        raise RuntimeError("NeuTTS is not installed. Run 'neutts install' first.")

    neu_module = importlib.import_module("neutts")
    NeuTTS = getattr(neu_module, "NeuTTS")

    config = AppConfig.load()
    tts = NeuTTS(
        backbone_repo=config.backbone_repo,
        backbone_device=config.backbone_device,
        codec_repo=config.codec_repo,
        codec_device=config.codec_device,
    )
    ref_codes = tts.encode_reference(ref_audio)
    wav = tts.infer(text, ref_codes, ref_text)
    return write_wav(out, wav, config.sample_rate)
