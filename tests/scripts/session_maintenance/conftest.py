"""The operator scripts are executable siblings, not an installed package."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'session_maintenance'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
