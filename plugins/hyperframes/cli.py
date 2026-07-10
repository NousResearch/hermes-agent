"""CLI for the hyperframes Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="hyperframes_command")

    subs.add_parser("status", help="Show skill link, CLI, FFmpeg, and preview readiness")

    setup = subs.add_parser("setup", help="Install npm hyperframes CLI and run doctor")
    setup.add_argument("--skip-chrome", action="store_true")

    install = subs.add_parser(
        "install",
        help="Link bundled skill, run setup, and optionally clone upstream examples",
    )
    install.add_argument("--force", action="store_true")
    install.add_argument("--ref", default="")
    install.add_argument("--skip-vendor", action="store_true")
    install.add_argument("--no-auto-prereqs", action="store_true")

    init = subs.add_parser("init", help="Scaffold a HyperFrames project")
    init.add_argument("project_name")
    init.add_argument("--project-dir", default="")
    init.add_argument("--example", default="")
    init.add_argument("--video", default="")
    init.add_argument("--audio", default="")
    init.add_argument("--interactive", action="store_true")

    validate = subs.add_parser("validate", aliases=["lint"], help="Lint, contrast-validate, and inspect")
    validate.add_argument("project_dir")
    validate.add_argument("--no-lint", action="store_true")
    validate.add_argument("--no-contrast", action="store_true")
    validate.add_argument("--no-inspect", action="store_true")
    validate.add_argument("--strict", action="store_true")

    render = subs.add_parser("render", help="Render a project to MP4/WebM")
    render.add_argument("project_dir")
    render.add_argument("--output", default="")
    render.add_argument("--quality", choices=["draft", "standard", "high"], default="standard")
    render.add_argument("--fps", type=int, default=None)
    render.add_argument("--format", choices=["mp4", "webm"], default=None)
    render.add_argument("--strict", action="store_true")

    preview = subs.add_parser("preview", help="Start or stop live preview")
    preview.add_argument("action", choices=["start", "stop"])
    preview.add_argument("project_dir", nargs="?", default="")
    preview.add_argument("--port", type=int, default=None)
    preview.add_argument("--force", action="store_true")

    capture = subs.add_parser("capture", help="Capture a website into HyperFrames assets")
    capture.add_argument("url")
    capture.add_argument("--output-dir", default="")
    capture.add_argument("--skip-assets", action="store_true")
    capture.add_argument("--json", dest="json_output", action="store_true")

    audio = subs.add_parser("audio", help="TTS narration or transcribe captions")
    audio.add_argument("action", choices=["tts", "transcribe"])
    audio.add_argument("--text", default="")
    audio.add_argument("--input", dest="input_path", default="")
    audio.add_argument("--output", dest="output_path", default="")
    audio.add_argument("--voice", default="")
    audio.add_argument("--lang", default="")

    subparser.set_defaults(func=hyperframes_command)


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok", True) else 1


def hyperframes_command(args: argparse.Namespace) -> int:
    command = getattr(args, "hyperframes_command", None)
    if not command:
        print(
            "usage: hermes hyperframes "
            "{status,setup,install,init,validate,render,preview,capture,audio}"
        )
        return 2

    if command == "status":
        return _print(core.status())
    if command == "setup":
        return _print(core.setup_environment(skip_chrome=getattr(args, "skip_chrome", False)))
    if command == "install":
        ref = (getattr(args, "ref", "") or "").strip() or None
        return _print(
            core.install(
                force=getattr(args, "force", False),
                ref=ref,
                skip_vendor=getattr(args, "skip_vendor", False),
                auto_prereqs=not getattr(args, "no_auto_prereqs", False),
            )
        )
    if command == "init":
        return _print(
            core.init_project(
                project_name=args.project_name,
                project_dir=getattr(args, "project_dir", "") or None,
                example=getattr(args, "example", "") or None,
                video=getattr(args, "video", "") or None,
                audio=getattr(args, "audio", "") or None,
                non_interactive=not getattr(args, "interactive", False),
            )
        )
    if command in {"validate", "lint"}:
        return _print(
            core.validate_project(
                project_dir=args.project_dir,
                lint=not getattr(args, "no_lint", False),
                contrast=not getattr(args, "no_contrast", False),
                inspect_layout=not getattr(args, "no_inspect", False),
                strict=getattr(args, "strict", False),
            )
        )
    if command == "render":
        return _print(
            core.render_project(
                project_dir=args.project_dir,
                output=getattr(args, "output", "") or None,
                quality=getattr(args, "quality", "standard"),
                fps=getattr(args, "fps", None),
                format=getattr(args, "format", None),
                strict=getattr(args, "strict", False),
            )
        )
    if command == "preview":
        return _print(
            core.preview_control(
                action=args.action,
                project_dir=getattr(args, "project_dir", "") or None,
                port=getattr(args, "port", None),
                force=getattr(args, "force", False),
            )
        )
    if command == "capture":
        return _print(
            core.capture_url(
                url=args.url,
                output_dir=getattr(args, "output_dir", "") or None,
                skip_assets=getattr(args, "skip_assets", False),
                json_output=getattr(args, "json_output", False),
            )
        )
    if command == "audio":
        return _print(
            core.audio_command(
                action=args.action,
                text=getattr(args, "text", "") or None,
                input_path=getattr(args, "input_path", "") or None,
                output_path=getattr(args, "output_path", "") or None,
                voice=getattr(args, "voice", "") or None,
                lang=getattr(args, "lang", "") or None,
            )
        )
    print(f"unknown hyperframes subcommand: {command}")
    return 2
