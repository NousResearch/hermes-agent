"""CLI command for the AITuber OnAir Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="aituber_onair_command")

    configure = subs.add_parser(
        "configure", aliases=["setup"], help="Save Hakua bridge settings"
    )
    configure.add_argument("--repo-root", default="")
    configure.add_argument("--model", default="")
    configure.add_argument(
        "--reply-backend", choices=["auto", "hermes", "codex"], default=""
    )
    configure.add_argument("--hermes-provider", default="")
    configure.add_argument("--hermes-model", default="")
    configure.add_argument("--fbx-port", type=int, default=None)
    configure.add_argument("--vrm-port", type=int, default=None)
    configure.add_argument("--avatar", choices=["fbx", "vrm", "vroid"], default="")
    configure.add_argument("--avatar-host", default="")
    configure.add_argument("--avatar-public-host", default="")
    configure.add_argument("--system-prompt", default="")
    configure.add_argument(
        "--tts-provider", choices=["auto", "irodori", "voicevox", "none"], default=""
    )
    configure.add_argument("--voicevox-url", default="")
    configure.add_argument("--voicevox-speaker", type=int, default=None)
    configure.add_argument("--voicevox-engine-exe", default="")
    configure.add_argument("--tts-voice", default="")
    configure.add_argument("--tts-speed", type=float, default=None)
    configure.add_argument("--youtube-live-id", default="")
    configure.add_argument("--stream-url", default="")
    configure.add_argument("--with-runtime-context", action="store_true", default=None)

    subs.add_parser("status", help="Show AITuber OnAir bridge readiness")
    context_status = subs.add_parser(
        "context-status",
        aliases=["runtime-context"],
        help="Validate safe memory/env context for Hakua speech",
    )
    context_status.add_argument("--prompt", default="")
    context_status.add_argument("--url", default="")

    stream_tweet = subs.add_parser(
        "stream-start-tweet",
        aliases=["tweet-start", "post-start"],
        help="Draft or publish a URL-bearing stream-start post via lm-twitterer",
    )
    stream_tweet.add_argument("--url", default="")
    stream_tweet.add_argument("--text", default="")
    stream_tweet.add_argument("--topic", default="")
    stream_tweet.add_argument("--live", action="store_true")
    stream_tweet.add_argument("--allow-private-url", action="store_true")
    stream_tweet.add_argument("--provider", default="")
    stream_tweet.add_argument("--model", default="")

    prepare = subs.add_parser("prepare", help="Prepare Codex SDK character chat")
    prepare.add_argument("--repo-root", default="")
    prepare.add_argument("--no-install-codex-sdk", action="store_true")
    prepare.add_argument("--no-build-chat", action="store_true")
    prepare.add_argument("--build-fbx-app", action="store_true")
    prepare.add_argument("--build-vrm-app", action="store_true")
    prepare.add_argument("--timeout-seconds", type=int, default=None)

    start = subs.add_parser("start", help="Start the avatar React app")
    start.add_argument("--repo-root", default="")
    start.add_argument("--avatar", choices=["fbx", "vrm", "vroid"], default="")
    start.add_argument("--fbx-port", type=int, default=None)
    start.add_argument("--vrm-port", type=int, default=None)
    start.add_argument("--host", default="")
    start.add_argument("--public-host", default="")
    start.add_argument("--force", action="store_true")

    galaxy = subs.add_parser(
        "galaxy-session",
        aliases=["galaxy", "session"],
        help="Start, stop, or inspect a Galaxy S9+ VRM session",
    )
    galaxy.add_argument(
        "--action",
        choices=["start", "status", "stop", "restart"],
        default="start",
    )
    galaxy.add_argument("--repo-root", default="")
    galaxy.add_argument("--vrm-port", type=int, default=None)
    galaxy.add_argument("--public-host", default="")
    galaxy.add_argument("--audio-ws-port", type=int, default=None)
    galaxy.add_argument("--force", action="store_true")
    galaxy.add_argument("--start-tts", action="store_true")
    galaxy.add_argument("--start-autonomous", action="store_true")
    galaxy.add_argument("--start-comment-reactions", action="store_true")
    galaxy.add_argument("--topic", default="")
    galaxy.add_argument("--interval-seconds", type=float, default=None)
    galaxy.add_argument("--poll-seconds", type=float, default=None)
    galaxy.add_argument("--play", action="store_true")
    galaxy.add_argument("--no-stop-loops", action="store_true")

    stop = subs.add_parser("stop", help="Stop the plugin-managed avatar React app")
    stop.add_argument("--force", action="store_true")

    subs.add_parser(
        "tts-status", aliases=["tts"], help="Show local Hakua TTS readiness"
    )

    start_tts = subs.add_parser(
        "start-tts", help="Start the selected local Hakua TTS backend"
    )
    start_tts.add_argument(
        "--provider", choices=["auto", "irodori", "voicevox"], default=""
    )
    start_tts.add_argument("--timeout-seconds", type=int, default=None)
    start_tts.add_argument("--voicevox-url", default="")
    start_tts.add_argument("--voicevox-speaker", type=int, default=None)

    speak = subs.add_parser("speak", help="Synthesize Hakua speech through local TTS")
    speak.add_argument("text", nargs="*")
    speak.add_argument(
        "--provider", choices=["auto", "irodori", "voicevox"], default=""
    )
    speak.add_argument("--output-path", default="")
    speak.add_argument("--format", default="")
    speak.add_argument("--voice", default="")
    speak.add_argument("--model", default="")
    speak.add_argument("--speed", type=float, default=None)
    speak.add_argument("--voicevox-speaker", type=int, default=None)
    speak.add_argument("--play", action="store_true")

    say = subs.add_parser("say", help="Ask Hakua to reply once")
    say.add_argument("prompt", nargs="*")
    say.add_argument("--repo-root", default="")
    say.add_argument("--model", default="")
    say.add_argument("--reply-backend", choices=["auto", "hermes", "codex"], default="")
    say.add_argument("--hermes-provider", default="")
    say.add_argument("--hermes-model", default="")
    say.add_argument("--response-length", default="")
    say.add_argument("--timeout-seconds", type=int, default=None)
    say.add_argument("--speak", action="store_true")
    say.add_argument(
        "--tts-provider", choices=["auto", "irodori", "voicevox"], default=""
    )
    say.add_argument("--tts-voice", default="")
    say.add_argument("--tts-speed", type=float, default=None)
    say.add_argument("--output-path", default="")
    say.add_argument("--play", action="store_true")
    say.add_argument("--with-runtime-context", action="store_true")

    smoke = subs.add_parser("smoke", help="Run a short Hakua readiness prompt")
    smoke.add_argument("--repo-root", default="")
    smoke.add_argument(
        "--reply-backend", choices=["auto", "hermes", "codex"], default=""
    )
    smoke.add_argument("--hermes-provider", default="")
    smoke.add_argument("--hermes-model", default="")
    smoke.add_argument("--timeout-seconds", type=int, default=None)

    youtube_ready = subs.add_parser(
        "youtube-ready",
        aliases=["youtube", "onair-ready"],
        help="Check OBS and YouTube encoder readiness for Hakua",
    )
    youtube_ready.add_argument("--no-require-obs", action="store_true")
    youtube_ready.add_argument("--require-tts-ready", action="store_true")

    start_comments = subs.add_parser(
        "start-comments",
        aliases=["comments-start", "onair-comments"],
        help="Start Hermes-side YouTube Live comment reactions",
    )
    start_comments.add_argument("--live-id", default="")
    start_comments.add_argument("--api-key-env", default="")
    start_comments.add_argument("--poll-seconds", type=float, default=None)
    start_comments.add_argument("--skip-existing", action="store_true")
    start_comments.add_argument("--no-play", action="store_true")
    start_comments.add_argument("--force", action="store_true")

    subs.add_parser(
        "comments-status",
        aliases=["comment-status"],
        help="Show YouTube Live comment monitor status",
    )

    stop_comments = subs.add_parser(
        "stop-comments",
        aliases=["comments-stop"],
        help="Stop Hermes-side YouTube Live comment reactions",
    )
    stop_comments.add_argument("--force", action="store_true")

    start_autonomous = subs.add_parser(
        "start-autonomous",
        aliases=["autonomous-start", "idle-talk"],
        help="Start Hakua autonomous idle talk",
    )
    start_autonomous.add_argument("--interval-seconds", type=float, default=None)
    start_autonomous.add_argument("--topic", default="")
    start_autonomous.add_argument("--no-play", action="store_true")
    start_autonomous.add_argument("--force", action="store_true")

    start_reactions = subs.add_parser(
        "start-reactions",
        aliases=["reactions-start", "start-local-comments"],
        help="Start Hakua local comment reaction loop",
    )
    start_reactions.add_argument("--poll-seconds", type=float, default=None)
    start_reactions.add_argument("--no-play", action="store_true")
    start_reactions.add_argument("--force", action="store_true")

    enqueue = subs.add_parser(
        "comment",
        aliases=["enqueue-comment"],
        help="Append a local comment for Hakua to react to",
    )
    enqueue.add_argument("text", nargs="*")
    enqueue.add_argument("--author", default="")
    enqueue.add_argument("--source", default="")

    subs.add_parser(
        "loops-status",
        aliases=["loop-status"],
        help="Show local autonomous/comment reaction loop status",
    )

    stop_loops = subs.add_parser(
        "stop-loops",
        aliases=["loops-stop"],
        help="Stop local autonomous/comment reaction loops",
    )
    stop_loops.add_argument(
        "--target", choices=["all", "autonomous", "comments"], default="all"
    )
    stop_loops.add_argument("--force", action="store_true")

    subparser.set_defaults(func=aituber_onair_command)


def aituber_onair_command(args: argparse.Namespace) -> int:
    command = getattr(args, "aituber_onair_command", None)
    if not command:
        print(
            "usage: hermes aituber-onair "
            "{configure,status,context-status,stream-start-tweet,prepare,start,galaxy-session,stop,tts-status,start-tts,speak,say,smoke,youtube-ready,start-comments,comments-status,stop-comments,start-autonomous,start-reactions,comment,loops-status,stop-loops}"
        )
        return 2
    if command in {"configure", "setup"}:
        return _print(
            core.save_hakua_config({
                "repo_root": getattr(args, "repo_root", ""),
                "model": getattr(args, "model", ""),
                "reply_backend": getattr(args, "reply_backend", ""),
                "hermes_provider": getattr(args, "hermes_provider", ""),
                "hermes_model": getattr(args, "hermes_model", ""),
                "fbx_port": getattr(args, "fbx_port", None),
                "vrm_port": getattr(args, "vrm_port", None),
                "avatar_kind": getattr(args, "avatar", ""),
                "avatar_host": getattr(args, "avatar_host", ""),
                "avatar_public_host": getattr(args, "avatar_public_host", ""),
                "system_prompt": getattr(args, "system_prompt", ""),
                "tts_provider": getattr(args, "tts_provider", ""),
                "voicevox_url": getattr(args, "voicevox_url", ""),
                "voicevox_speaker": getattr(args, "voicevox_speaker", None),
                "voicevox_engine_exe": getattr(args, "voicevox_engine_exe", ""),
                "tts_voice": getattr(args, "tts_voice", ""),
                "tts_speed": getattr(args, "tts_speed", None),
                "youtube_live_id": getattr(args, "youtube_live_id", ""),
                "stream_url": getattr(args, "stream_url", ""),
                "with_runtime_context": getattr(args, "with_runtime_context", False),
            })
        )
    if command == "status":
        return _print(core.status())
    if command in {"context-status", "runtime-context"}:
        return _print(
            core.context_status({
                "prompt": getattr(args, "prompt", ""),
                "url": getattr(args, "url", ""),
            })
        )
    if command in {"stream-start-tweet", "tweet-start", "post-start"}:
        return _print(
            core.stream_start_tweet({
                "url": getattr(args, "url", ""),
                "text": getattr(args, "text", ""),
                "topic": getattr(args, "topic", ""),
                "live": getattr(args, "live", False),
                "allow_private_url": getattr(args, "allow_private_url", False),
                "provider": getattr(args, "provider", ""),
                "model": getattr(args, "model", ""),
            })
        )
    if command == "prepare":
        return _print(
            core.prepare({
                "repo_root": getattr(args, "repo_root", ""),
                "install_codex_sdk": not getattr(args, "no_install_codex_sdk", False),
                "build_chat": not getattr(args, "no_build_chat", False),
                "build_fbx_app": getattr(args, "build_fbx_app", False),
                "build_vrm_app": getattr(args, "build_vrm_app", False),
                "timeout_seconds": getattr(args, "timeout_seconds", None),
            })
        )
    if command == "start":
        return _print(
            core.start_avatar_app({
                "repo_root": getattr(args, "repo_root", ""),
                "avatar_kind": getattr(args, "avatar", ""),
                "fbx_port": getattr(args, "fbx_port", None),
                "vrm_port": getattr(args, "vrm_port", None),
                "host": getattr(args, "host", ""),
                "public_host": getattr(args, "public_host", ""),
                "force": getattr(args, "force", False),
            })
        )
    if command in {"galaxy-session", "galaxy", "session"}:
        return _print(
            core.galaxy_session({
                "action": getattr(args, "action", "start"),
                "repo_root": getattr(args, "repo_root", ""),
                "vrm_port": getattr(args, "vrm_port", None),
                "public_host": getattr(args, "public_host", ""),
                "audio_ws_port": getattr(args, "audio_ws_port", None),
                "force": getattr(args, "force", False),
                "start_tts": getattr(args, "start_tts", False),
                "start_autonomous": getattr(args, "start_autonomous", False),
                "start_comment_reactions": getattr(
                    args, "start_comment_reactions", False
                ),
                "topic": getattr(args, "topic", ""),
                "interval_seconds": getattr(args, "interval_seconds", None),
                "poll_seconds": getattr(args, "poll_seconds", None),
                "play": getattr(args, "play", False),
                "stop_loops": not getattr(args, "no_stop_loops", False),
            })
        )
    if command == "stop":
        return _print(core.stop_fbx_app({"force": getattr(args, "force", False)}))
    if command in {"tts-status", "tts"}:
        return _print(core.tts_status())
    if command == "start-tts":
        return _print(
            core.start_tts({
                "provider": getattr(args, "provider", ""),
                "timeout_seconds": getattr(args, "timeout_seconds", None),
                "voicevox_url": getattr(args, "voicevox_url", ""),
                "voicevox_speaker": getattr(args, "voicevox_speaker", None),
            })
        )
    if command == "speak":
        return _print(
            core.synthesize_speech({
                "text": " ".join(getattr(args, "text", [])).strip(),
                "provider": getattr(args, "provider", ""),
                "output_path": getattr(args, "output_path", ""),
                "format": getattr(args, "format", ""),
                "voice": getattr(args, "voice", ""),
                "model": getattr(args, "model", ""),
                "speed": getattr(args, "speed", None),
                "voicevox_speaker": getattr(args, "voicevox_speaker", None),
                "play": getattr(args, "play", False),
            })
        )
    if command == "say":
        return _print(
            core.run_hakua_once({
                "prompt": " ".join(getattr(args, "prompt", [])).strip(),
                "repo_root": getattr(args, "repo_root", ""),
                "model": getattr(args, "model", ""),
                "reply_backend": getattr(args, "reply_backend", ""),
                "hermes_provider": getattr(args, "hermes_provider", ""),
                "hermes_model": getattr(args, "hermes_model", ""),
                "response_length": getattr(args, "response_length", ""),
                "timeout_seconds": getattr(args, "timeout_seconds", None),
                "speak": getattr(args, "speak", False),
                "tts_provider": getattr(args, "tts_provider", ""),
                "tts_voice": getattr(args, "tts_voice", ""),
                "tts_speed": getattr(args, "tts_speed", None),
                "output_path": getattr(args, "output_path", ""),
                "play": getattr(args, "play", False),
                "with_runtime_context": getattr(args, "with_runtime_context", False),
            })
        )
    if command == "smoke":
        payload = json.loads(
            core.handle_smoke({
                "repo_root": getattr(args, "repo_root", ""),
                "reply_backend": getattr(args, "reply_backend", ""),
                "hermes_provider": getattr(args, "hermes_provider", ""),
                "hermes_model": getattr(args, "hermes_model", ""),
                "timeout_seconds": getattr(args, "timeout_seconds", None),
            })
        )
        return _print(payload)
    if command in {"youtube-ready", "youtube", "onair-ready"}:
        return _print(
            core.youtube_ready({
                "require_obs": not getattr(args, "no_require_obs", False),
                "require_tts_ready": getattr(args, "require_tts_ready", False),
            })
        )
    if command in {"start-comments", "comments-start", "onair-comments"}:
        return _print(
            core.start_youtube_comments({
                "live_id": getattr(args, "live_id", ""),
                "api_key_env": getattr(args, "api_key_env", ""),
                "poll_seconds": getattr(args, "poll_seconds", None),
                "skip_existing": getattr(args, "skip_existing", False),
                "play": not getattr(args, "no_play", False),
                "force": getattr(args, "force", False),
            })
        )
    if command in {"comments-status", "comment-status"}:
        return _print(core.youtube_comments_status({}))
    if command in {"stop-comments", "comments-stop"}:
        return _print(
            core.stop_youtube_comments({"force": getattr(args, "force", False)})
        )
    if command in {"start-autonomous", "autonomous-start", "idle-talk"}:
        return _print(
            core.start_autonomous_talk_loop({
                "interval_seconds": getattr(args, "interval_seconds", None),
                "topic": getattr(args, "topic", ""),
                "play": not getattr(args, "no_play", False),
                "force": getattr(args, "force", False),
            })
        )
    if command in {"start-reactions", "reactions-start", "start-local-comments"}:
        return _print(
            core.start_comment_reaction_loop({
                "poll_seconds": getattr(args, "poll_seconds", None),
                "play": not getattr(args, "no_play", False),
                "force": getattr(args, "force", False),
            })
        )
    if command in {"comment", "enqueue-comment"}:
        return _print(
            core.enqueue_comment({
                "text": " ".join(getattr(args, "text", [])).strip(),
                "author": getattr(args, "author", ""),
                "source": getattr(args, "source", ""),
            })
        )
    if command in {"loops-status", "loop-status"}:
        return _print(core.loops_status({}))
    if command in {"stop-loops", "loops-stop"}:
        return _print(
            core.stop_loops({
                "target": getattr(args, "target", "all"),
                "force": getattr(args, "force", False),
            })
        )
    print("unknown aituber-onair command")
    return 2


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok") else 1
