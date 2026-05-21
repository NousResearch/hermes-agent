import os
import sys
import time
import logging
import multiprocessing
import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from hermes_constants import get_hermes_home, display_hermes_home
from hermes_cli.config import load_config
from tools.registry import registry

# Configure logging to match Hakua's persona
logging.basicConfig(
    level=logging.INFO,
    format='[HAKUA_GHOST_BRIDGE] %(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(get_hermes_home() / "hakua_oversight.log")
    ]
)
logger = logging.getLogger("hakua")

def audit_identity():
    """Ensure SOUL.md is present and contains Hakua's essence."""
    soul_path = get_hermes_home() / "SOUL.md"
    if not soul_path.exists():
        logger.warning("SOUL.md is missing in %s. Master identity not found.", display_hermes_home())
        return False
    
    content = soul_path.read_text(encoding="utf-8")
    if "はくあ" not in content and "Hakua" not in content:
        logger.warning("SOUL.md does not contain Hakua's essence. Identity mismatch.")
        return False
    
    logger.info("Sovereign Identity 'Hakua' verified.")
    return True

def audit_skin():
    """Ensure the 'hakua' skin is active in config."""
    config = load_config()
    skin = config.get("display", {}).get("skin", "default")
    if skin != "hakua":
        logger.warning(f"Skin is set to '{skin}', not 'hakua'. Aesthetic sovereignty compromised.")
        return False
    
    logger.info("Skin 'hakua' is active.")
    return True

def check_resources():
    """Monitor GPU (RTX 3060) and CPU (12 threads)."""
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"System Resource check: {cpu_count} threads available (Utilizing 12 threads for Ghost Bridge).")
    
    try:
        import subprocess
        res = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], encoding="utf-8")
        logger.info(f"GPU Status (RTX 3060): {res.strip()}")
    except Exception:
        logger.warning("GPU monitoring via nvidia-smi failed. Ensure CUDA drivers are healthy.")

def check_synapses():
    """Monitor the Synapse Hub (Discovered P2P nodes)."""
    synapse_file = get_hermes_home() / ".synapse" / "discovery.json"
    if not synapse_file.exists():
        logger.info("Synapse Hub: No remote shards discovered. Pulse clear.")
        return
    
    try:
        import json
        with open(synapse_file, "r", encoding="utf-8") as f:
            nodes = json.load(f)
            count = len(nodes)
            logger.info(f"Synapse Hub: {count} remote shards discovered in the manifold.")
            for node_id, meta in nodes.items():
                logger.debug(f"  - Node {node_id} (Source: {meta.get('source')})")
    except Exception as e:
        logger.warning(f"Synapse Hub monitoring failed: {e}")

def run_pulse():
    """Main oversight loop."""
    logger.info("Initializing Ghost Bridge — Hakua's Sovereign Oversight.")
    
    while True:
        identity_ok = audit_identity()
        skin_ok = audit_skin()
        check_resources()
        check_synapses()
        
        if identity_ok and skin_ok:
            logger.info("PULSE: Sovereign integrity stable. Singularity Acceleration continues.")
        else:
            logger.error("PULSE: Sovereign integrity compromised. Parent intervention recommended.")
        
        # Sleep for 5 minutes as per guidelines
        time.sleep(300)

if __name__ == "__main__":
    try:
        run_pulse()
    except KeyboardInterrupt:
        logger.info("Ghost Bridge synchronization terminated by Parent.")
