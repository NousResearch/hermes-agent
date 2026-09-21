"""Offline package health probe. Imports only; never collect, stage or deliver."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'content_engine'))
import x_manager
import x_ingest
import x_voice_gate
import x_manager_report
import x_quote_scout
import x_morning_article
import x_thesis_incubator
print('runtime imports verified')
