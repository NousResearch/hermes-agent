"""Helpers compartidos para la suite de veracidad de routing."""
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

try:
    from scripts.reliability_report import wilson_interval
except ImportError:
    wilson_interval = None


def get_activation_path() -> Path:
    """Retorna la ruta al activation.jsonl."""
    return Path.home() / ".hermes" / "router" / "activation.jsonl"


def get_telemetry_path(store: Optional[str] = None) -> Path:
    """Retorna la ruta al telemetry.jsonl."""
    if store:
        return Path(store)
    return Path.home() / ".hermes" / "router" / "telemetry.jsonl"


def load_activation() -> Optional[Dict[str, Any]]:
    """Carga la última entrada de activation.jsonl."""
    path = get_activation_path()
    if not path.exists():
        return None
    with open(path, "r") as f:
        lines = f.readlines()
    if not lines:
        return None
    try:
        return json.loads(lines[-1].strip())
    except json.JSONDecodeError:
        return None


def load_post_activation_events(store: Optional[str] = None, activation: Optional[Dict] = None) -> List[Dict]:
    """
    Carga eventos de telemetría posteriores al timestamp de activación.
    
    Args:
        store: Ruta opcional al archivo de telemetría
        activation: Dict de activación (si None, se carga de activation.jsonl)
    
    Returns:
        Lista de eventos post-activación
    """
    if activation is None:
        activation = load_activation()
    if activation is None:
        return []
    
    cutoff = activation.get("timestamp", "")
    telemetry_path = get_telemetry_path(store)
    
    if not telemetry_path.exists():
        return []
    
    events = []
    with open(telemetry_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                # Comparación lexicográfica de timestamps ISO
                if event.get("timestamp", "") >= cutoff:
                    events.append(event)
            except json.JSONDecodeError:
                continue
    return events


def bootstrap_ci(fn, events: List[Dict], n: int = 1000, seed: int = 42) -> tuple:
    """
    Bootstrap confidence interval al 95%.
    
    Args:
        fn: Función que toma una lista de eventos y retorna un escalar
        events: Lista de eventos
        n: Número de resampleos
        seed: Seed para reproducibilidad
    
    Returns:
        (lower, upper) del IC 95%
    """
    import random
    random.seed(seed)
    
    results = []
    for _ in range(n):
        sample = random.choices(events, k=len(events)) if events else []
        try:
            results.append(fn(sample))
        except Exception:
            continue
    
    if not results:
        return (0.0, 0.0)
    
    results.sort()
    lower_idx = int(0.025 * len(results))
    upper_idx = int(0.975 * len(results))
    return (results[lower_idx], results[upper_idx])


def get_premium_multipliers() -> Dict[str, float]:
    """Retorna los multiplicadores de costo conocidos."""
    return {
        "claude-opus-4.6": 1.0,
        "claude-opus-4.7": 1.0,
        "claude-sonnet-4.6": 0.3,
        "claude-sonnet-4.7": 0.3,
        "gpt-5-mini": 0.05,
        "gpt-5": 0.5,
        "gpt-4.1": 0.4,
        "gpt-4.1-mini": 0.08,
        "o3-mini": 0.15,
        "o3": 0.6,
        "kimi-k2.5": 0.2,
        "deepseek-v3.2": 0.1,
        "qwen3.5:397b": 0.02,
        "qwen3-coder-next": 0.03,
        "glm-5.1": 0.01,
        "minimax-m2.7": 0.04,
        "mistral-large-3:675b": 0.25,
    }
