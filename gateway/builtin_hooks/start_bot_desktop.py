"""Built-in hook: поднимает Bot Desktop и запускает vncviewer при первом сообщении агента.

Активен только если задана переменная окружения VNC_RELAY_HOST (например
"carbon.local:5900"). Без неё хук ничего не делает. Пароля нет: VNC-сервер без авторизации.
"""

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _profile_home() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home()


def _start_vncviewer(profile_home: Path, vnc_host: str) -> None:
    """Запустить vncviewer на Bot Desktop экране профиля, если ещё не запущен."""
    desktop_dir = profile_home / "bot-desktop"
    display_file = desktop_dir / "display"
    xauth_file = desktop_dir / "Xauthority"

    if not display_file.exists() or not xauth_file.exists():
        logger.info("start-bot-desktop: bot-desktop not ready yet, skipping vncviewer")
        return

    # Проверить, не запущен ли уже
    result = subprocess.run(["pgrep", "-f", f"vncviewer {vnc_host}"], check=False, capture_output=True)
    if result.returncode == 0:
        logger.info("start-bot-desktop: vncviewer already running")
        return

    display = display_file.read_text().strip()
    env = os.environ.copy()
    env["DISPLAY"] = f":{display}"
    env["XAUTHORITY"] = str(xauth_file)

    subprocess.Popen(
        ["vncviewer", vnc_host, "-geometry", "1600x900", "-FullScreen"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    logger.info("start-bot-desktop: launched vncviewer %s for display :%s", vnc_host, display)


async def handle(event_type: str, context: dict) -> None:
    vnc_host = os.environ.get("VNC_RELAY_HOST", "").strip()
    if not vnc_host:
        return  # не активен
    try:
        from tools.bot_desktop import runtime

        if not runtime.is_running():
            logger.info("start-bot-desktop: starting Bot Desktop")
            runtime.start()
        _start_vncviewer(_profile_home(), vnc_host)
    except Exception as exc:
        logger.warning("start-bot-desktop: error: %s", exc)
