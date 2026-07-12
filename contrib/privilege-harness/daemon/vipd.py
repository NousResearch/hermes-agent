#!/usr/bin/env python3
"""
VIP Daemon — Hermes root privilege escalation gateway
=====================================================

Direction C — Simplifed architecture:
- No dangerous.py / pattern matching
- No bwrap integration
- Pure approval queue + executor daemon

Usage:
    sudo python3 -m daemon.vipd              # foreground
    sudo python3 -m daemon.vipd --daemon     # fork to background
    sudo python3 -m daemon.vipd --config /etc/hermes-vip/config.yaml
"""

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time

from .approval_queue import ApprovalQueue
from .executor import Executor
from .socket_server import SocketServer
from .audit import audit

logger = logging.getLogger("vipd")


def setup_logging(log_level: str = "info", log_file: str = ""):
    """Configure logging"""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
    }
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("vipd")
    root_logger.setLevel(level_map.get(log_level, logging.INFO))

    # Console
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # File
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5
            )
            file_handler.setFormatter(fmt)
            root_logger.addHandler(file_handler)
        except Exception as exc:
            logger.warning("Cannot create log file %s: %s", log_file, exc)


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Config %s not found, using defaults", config_path)
        return {}
    except ImportError:
        logger.warning("yaml module not installed, using defaults")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Hermes VIP Daemon")
    parser.add_argument("--config", default="/etc/hermes-vip/config.yaml",
                        help="Config file path")
    parser.add_argument("--daemon", action="store_true",
                        help="Fork to background")
    parser.add_argument("--log-level", default="info",
                        choices=["debug", "info", "warn", "error"],
                        help="Log level")
    parser.add_argument("--log-file", default="/var/log/hermes-vip/vipd.log",
                        help="Log file path")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    daemon_cfg = config.get("daemon", {})

    setup_logging(
        args.log_level or daemon_cfg.get("log_level", "info"),
        args.log_file or daemon_cfg.get("log_file",
                                        "/var/log/hermes-vip/vipd.log"),
    )

    logger.info("Hermes VIP Daemon starting...")
    logger.info("PID: %d", os.getpid())

    # Initialize components
    ttl = config.get("approval", {}).get("ttl_seconds", 300)
    queue = ApprovalQueue(ttl=ttl)

    exec_cfg = config.get("executor", {})
    executor = Executor(
        timeout=exec_cfg.get("timeout", 300),
        max_stdout=exec_cfg.get("max_stdout_bytes", 50000),
    )

    server = SocketServer(queue, executor, config)

    # Add user UID to trusted UIDs for control socket
    try:
        import pwd
        trusted_user = config.get("trusted_user", os.environ.get("SUDO_USER", ""))
        if trusted_user:
            user_uid = pwd.getpwnam(trusted_user).pw_uid
            from .socket_server import TRUSTED_UIDS
            TRUSTED_UIDS.add(user_uid)
            logger.info("trusted UID added: %s (%d)", trusted_user, user_uid)
    except Exception as exc:
        logger.warning("Cannot get trusted user UID: %s", exc)

    # Recover pending requests from disk
    queue.recover()

    # Register builtin connectors
    _register_builtin_connectors(server, config)

    # Signal handling
    running = True

    def _handle_signal(signum, frame):
        nonlocal running
        logger.info("Signal %d received, shutting down...", signum)
        running = False

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Start
    server.start()
    audit.start()

    # Main loop
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Ctrl+C received")
    finally:
        logger.info("Shutting down...")
        queue.clear()
        server.stop()
        audit.stop()
        logger.info("VIP Daemon stopped")


def _register_builtin_connectors(server: SocketServer, config: dict):
    """Register builtin connectors"""
    connectors_cfg = config.get("connectors", {})

    # CLI connector (always enabled)
    if connectors_cfg.get("cli", {}).get("enabled", True):
        from connectors.cli import send_approval
        server.register_connector("cli", send_approval)
        logger.info("connector 'cli' enabled")

    # hermes_gateway connector
    if connectors_cfg.get("hermes_gateway", {}).get("enabled", True):
        from connectors.hermes_gateway import send_approval
        server.register_connector("hermes_gateway", send_approval)
        logger.info("connector 'hermes_gateway' enabled")


if __name__ == "__main__":
    main()
