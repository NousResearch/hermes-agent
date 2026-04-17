"""Tests para el activation marker."""
import json
import os
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.routing.veracity._helpers import get_activation_path, load_activation


class TestActivationMarker:
    """Valida que el activation.jsonl existe y es válido."""

    def test_activation_jsonl_exists_and_is_valid(self):
        """activation.jsonl debe existir y cada línea debe ser JSON válido con las keys requeridas."""
        path = get_activation_path()
        
        # Skip graceful si no existe y la var env lo permite
        require_activation = os.environ.get("HERMES_VERACITY_REQUIRE_ACTIVATION", "0")
        if not path.exists():
            if require_activation != "1":
                pytest.skip("activation.jsonl no existe y HERMES_VERACITY_REQUIRE_ACTIVATION!=1")
            else:
                pytest.fail("activation.jsonl no existe con HERMES_VERACITY_REQUIRE_ACTIVATION=1")
        
        required_keys = {"timestamp", "commit_sha", "reason", "config_snapshot_hash"}
        
        with open(path, "r") as f:
            lines = f.readlines()
        
        assert len(lines) > 0, "activation.jsonl está vacío"
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Línea {i+1} no es JSON válido: {e}")
            
            for key in required_keys:
                assert key in entry, f"Línea {i+1} falta key '{key}'"

    def test_latest_activation_is_recent(self):
        """La última entrada debe ser ≤ 30 días."""
        activation = load_activation()
        
        if activation is None:
            require_activation = os.environ.get("HERMES_VERACITY_REQUIRE_ACTIVATION", "0")
            if require_activation != "1":
                pytest.skip("No hay activation para validar")
            else:
                pytest.fail("No hay activation con HERMES_VERACITY_REQUIRE_ACTIVATION=1")
        
        timestamp_str = activation.get("timestamp", "")
        try:
            # Parse ISO format con timezone
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1] + "+00:00"
            activation_time = datetime.fromisoformat(timestamp_str)
        except ValueError:
            pytest.fail(f"Timestamp inválido: {timestamp_str}")
        
        now = datetime.now(timezone.utc)
        age = now - activation_time
        
        assert age <= timedelta(days=30), f"Activation muy antiguo: {age.days} días"
