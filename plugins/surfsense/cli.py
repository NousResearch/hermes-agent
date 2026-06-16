"""CLI commands for the SurfSense Hermes plugin."""

from __future__ import annotations

import argparse
import getpass
import json

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="surfsense_command")

    subs.add_parser("status", help="Show SurfSense plugin readiness")

    setup = subs.add_parser("setup", help="Save SurfSense endpoint settings")
    setup.add_argument("--base-url", default="")
    setup.add_argument("--frontend-url", default="")
    setup.add_argument("--root", default="")

    login = subs.add_parser("login", help="Log in and save a SurfSense bearer token")
    login.add_argument("--username", required=True)
    login.add_argument("--password", default="")
    login.add_argument("--no-save", action="store_true")

    spaces = subs.add_parser("spaces", help="List SurfSense search spaces")
    spaces.add_argument("--owned-only", action="store_true")
    spaces.add_argument("--limit", type=int, default=None)

    upload = subs.add_parser("upload", help="Upload files into a SurfSense search space")
    upload.add_argument("--search-space-id", type=int, required=True)
    upload.add_argument("--use-vision-llm", action="store_true")
    upload.add_argument("--processing-mode", default="basic")
    upload.add_argument("paths", nargs="+")

    search = subs.add_parser("search", help="Search SurfSense documents")
    search.add_argument("--search-space-id", type=int, required=True)
    search.add_argument("--page-size", type=int, default=10)
    search.add_argument("query")

    ask = subs.add_parser("ask", help="Ask SurfSense against a search space")
    ask.add_argument("--search-space-id", type=int, required=True)
    ask.add_argument("--thread-id", type=int, default=None)
    ask.add_argument("--title", default="")
    ask.add_argument("query")

    video = subs.add_parser("video-plan", help="Create NotebookLM-style video artifacts")
    video.add_argument("--renderer", choices=["all", "manim", "heygen", "hyperframes"], default="all")
    video.add_argument("--output-dir", default="")
    video.add_argument("--duration-seconds", type=int, default=120)
    video.add_argument("--language", default="ja")
    video.add_argument("--style", default="swiss_pulse")
    video.add_argument("--llm-wiki-text", default="")
    video.add_argument("--codegraph-text", default="")
    video.add_argument("--sleep-text", default="")
    video.add_argument("--memory-text", default="")
    video.add_argument("--evidence-policy", choices=["strict", "balanced"], default="strict")
    video.add_argument("--voice-pipeline", choices=["none", "irodori_tts", "aituber_onair", "all"], default="none")
    video.add_argument("--audio-text", default="")
    video.add_argument("--audio-output-path", default="")
    video.add_argument("--video-input-path", default="")
    video.add_argument("--output-mp4-path", default="")
    video.add_argument("--tts-voice", default="")
    video.add_argument("--tts-model", default="")
    video.add_argument("--tts-speed", type=float, default=None)
    video.add_argument("--tts-format", choices=["wav", "mp3", "flac", "opus", "aac", "pcm"], default="wav")
    video.add_argument("--source-text", required=True)
    video.add_argument("topic")

    mux = subs.add_parser("video-mux", help="Mux a silent video and narration audio into MP4")
    mux.add_argument("--dry-run", action="store_true")
    mux.add_argument("video_path")
    mux.add_argument("audio_path")
    mux.add_argument("output_path")

    docker = subs.add_parser("docker", help="Run a safe SurfSense docker compose helper")
    docker.add_argument("action", choices=["ps", "pull", "up", "logs"], default="ps")

    subparser.set_defaults(func=surfsense_command)


def surfsense_command(args: argparse.Namespace) -> int:
    command = getattr(args, "surfsense_command", None)
    if not command:
        print("usage: hermes surfsense {status,setup,login,spaces,upload,search,ask,video-plan,video-mux,docker}")
        return 2
    if command == "status":
        return _print(core.status())
    if command == "setup":
        return _print(_setup_values(args))
    if command == "login":
        password = getattr(args, "password", "") or getpass.getpass("SurfSense password: ")
        return _print(
            core.login(
                username=getattr(args, "username", ""),
                password=password,
                save=not bool(getattr(args, "no_save", False)),
            )
        )
    if command == "spaces":
        return _print(
            core.list_searchspaces(
                owned_only=bool(getattr(args, "owned_only", False)),
                limit=getattr(args, "limit", None),
            )
        )
    if command == "upload":
        return _print(
            core.upload_files(
                paths=list(getattr(args, "paths", [])),
                search_space_id=int(getattr(args, "search_space_id", 0)),
                use_vision_llm=bool(getattr(args, "use_vision_llm", False)),
                processing_mode=getattr(args, "processing_mode", "basic"),
            )
        )
    if command == "search":
        return _print(
            core.search_documents(
                query=getattr(args, "query", ""),
                search_space_id=int(getattr(args, "search_space_id", 0)),
                page_size=int(getattr(args, "page_size", 10)),
            )
        )
    if command == "ask":
        return _print(
            core.ask(
                query=getattr(args, "query", ""),
                search_space_id=int(getattr(args, "search_space_id", 0)),
                thread_id=getattr(args, "thread_id", None),
                title=getattr(args, "title", "") or None,
            )
        )
    if command == "video-plan":
        return _print(
            core.video_plan(
                topic=getattr(args, "topic", ""),
                source_text=getattr(args, "source_text", ""),
                renderer=getattr(args, "renderer", "all"),
                output_dir=getattr(args, "output_dir", "") or None,
                duration_seconds=int(getattr(args, "duration_seconds", 120)),
                language=getattr(args, "language", "ja"),
                style=getattr(args, "style", "swiss_pulse"),
                llm_wiki_text=getattr(args, "llm_wiki_text", ""),
                codegraph_text=getattr(args, "codegraph_text", ""),
                sleep_text=getattr(args, "sleep_text", ""),
                memory_text=getattr(args, "memory_text", ""),
                evidence_policy=getattr(args, "evidence_policy", "strict"),
                voice_pipeline=getattr(args, "voice_pipeline", "none"),
                audio_text=getattr(args, "audio_text", ""),
                audio_output_path=getattr(args, "audio_output_path", "") or None,
                video_input_path=getattr(args, "video_input_path", "") or None,
                output_mp4_path=getattr(args, "output_mp4_path", "") or None,
                tts_voice=getattr(args, "tts_voice", ""),
                tts_model=getattr(args, "tts_model", ""),
                tts_speed=getattr(args, "tts_speed", None),
                tts_format=getattr(args, "tts_format", "wav"),
            )
        )
    if command == "video-mux":
        return _print(
            core.video_mux(
                video_path=getattr(args, "video_path", ""),
                audio_path=getattr(args, "audio_path", ""),
                output_path=getattr(args, "output_path", ""),
                dry_run=bool(getattr(args, "dry_run", False)),
            )
        )
    if command == "docker":
        return _print(core.docker_compose(action=getattr(args, "action", "ps")))
    print(f"unknown command: {command}")
    return 2


def _setup_values(args: argparse.Namespace) -> dict:
    if core.save_env_value is None:
        return {"ok": False, "error": "Hermes env writer is unavailable."}
    saved: list[str] = []
    values = {
        "SURFSENSE_BASE_URL": getattr(args, "base_url", ""),
        "SURFSENSE_FRONTEND_URL": getattr(args, "frontend_url", ""),
        "SURFSENSE_ROOT": getattr(args, "root", ""),
    }
    for key, value in values.items():
        clean = str(value or "").strip()
        if clean:
            core.save_env_value(key, clean)
            saved.append(key)
    return {"ok": True, "saved": saved, "status": core.status()}


def _print(data: dict) -> int:
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0 if data.get("ok") else 1
