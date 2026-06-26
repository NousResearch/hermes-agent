#!/usr/bin/env python3
"""Self-Learning Security Module for Hermes Agent."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class SecurityIncident:
    id: str
    timestamp: float
    incident_type: str
    severity: str
    description: str
    context: Dict[str, Any]
    user_feedback: Optional[str] = None
    learned_from: bool = False
    similarity_hash: str = ""


@dataclass
class ThreatPattern:
    pattern_id: str
    pattern_type: str
    pattern_data: str
    confidence: float
    hit_count: int = 0
    false_positive_count: int = 0
    last_seen: float = 0.0
    source: str = "static"


@dataclass  
class SecurityContext:
    session_id: str
    trust_level: float = 1.0
    risk_score: float = 0.0
    allowed_commands: Set[str] = field(default_factory=set)
    blocked_patterns: Set[str] = field(default_factory=set)
    learning_enabled: bool = True


class SelfLearningSecurity:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")), "security_learning.db")
        self.incidents = {}
        self.patterns = {}
        self.session_contexts = {}
        self.threat_intel_cache = {}
        self.threat_intel_last_update = 0.0
        self.command_baselines = defaultdict(list)
        self.anomaly_threshold = 2.5
        self._db_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._init_database()
        self._load_patterns()
        self._start_threat_intel_updater()

    def _init_database(self):
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS incidents (id TEXT PRIMARY KEY, timestamp REAL, incident_type TEXT, severity TEXT, description TEXT, context TEXT, user_feedback TEXT, learned_from INTEGER, similarity_hash TEXT);
                    CREATE TABLE IF NOT EXISTS patterns (pattern_id TEXT PRIMARY KEY, pattern_type TEXT, pattern_data TEXT, confidence REAL, hit_count INTEGER, false_positive_count INTEGER, last_seen REAL, source TEXT);
                    CREATE TABLE IF NOT EXISTS threat_intel (id TEXT PRIMARY KEY, source TEXT, data TEXT, fetched_at REAL, expires_at REAL);
                    CREATE INDEX IF NOT EXISTS idx_incidents_type ON incidents(incident_type);
                """)
        except Exception as e:
            logger.warning("DB init failed: %s", e)

    def _load_patterns(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                for row in conn.execute("SELECT * FROM patterns"):
                    self.patterns[row[0]] = ThreatPattern(*row)
        except Exception:
            pass

    def _start_threat_intel_updater(self):
        def loop():
            while True:
                try:
                    self._fetch_threat_intelligence()
                except Exception:
                    pass
                time.sleep(3600)
        threading.Thread(target=loop, daemon=True).start()

    def _fetch_threat_intelligence(self):
        now = time.time()
        if now - self.threat_intel_last_update < 3600:
            return
        try:
            import urllib.request
            req = urllib.request.Request("https://api.github.com/advisories?per_page=20", headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                self.threat_intel_cache["github"] = json.loads(r.read().decode())[:20]
            vulns = []
            for eco in ["npm", "pypi"]:
                try:
                    with urllib.request.urlopen(f"https://api.osv.dev/v1/list?ecosystem={eco}", timeout=10) as r:
                        vulns.extend(json.loads(r.read().decode()).get("vulns", [])[:10])
                except Exception:
                    pass
            self.threat_intel_cache["osv"] = vulns[:50]
            self.threat_intel_last_update = now
        except Exception:
            pass

    def analyze_command(self, command, context=None):
        context = context or {}
        session_id = context.get("session_id", "default")
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = SecurityContext(session_id=session_id)
        session_ctx = self.session_contexts[session_id]
        findings = []
        risk_score = 0.0
        findings.extend(self._check_learned_patterns(command))
        findings.extend(self._check_threat_intel(command))
        anomaly = self._detect_anomalies(command, session_ctx)
        if anomaly > self.anomaly_threshold:
            findings.append({"type": "behavioral_anomaly", "severity": "medium", "description": f"Anomaly score: {anomaly:.2f}", "confidence": min(anomaly/5.0, 1.0)})
            risk_score += anomaly * 0.1
        for f in findings:
            w = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.2}.get(f.get("severity", "low"), 0.2)
            risk_score += w * f.get("confidence", 0.5)
        risk_score = min(risk_score, 10.0)
        if findings:
            session_ctx.trust_level = max(0.0, session_ctx.trust_level - 0.05 * len(findings))
        rec = "block" if risk_score >= 8.0 else "warn" if risk_score >= 5.0 else "monitor" if risk_score >= 2.0 else "allow"
        return {"risk_score": risk_score, "findings": findings, "recommendation": rec, "trust_level": session_ctx.trust_level, "requires_approval": rec in ("block", "warn")}

    def _check_learned_patterns(self, command):
        findings = []
        with self._cache_lock:
            for p in self.patterns.values():
                if p.confidence < 0.5:
                    continue
                match = False
                if p.pattern_type == "regex":
                    try:
                        match = bool(re.search(p.pattern_data, command))
                    except re.error:
                        pass
                elif p.pattern_type == "substring":
                    match = p.pattern_data in command
                if match:
                    p.hit_count += 1
                    fp_ratio = p.false_positive_count / max(p.hit_count, 1)
                    adj_conf = p.confidence * (1.0 - fp_ratio * 0.5)
                    findings.append({"type": "learned_pattern", "pattern_id": p.pattern_id, "severity": "high" if adj_conf > 0.8 else "medium", "description": f"Matched {p.pattern_type}", "confidence": adj_conf, "source": "self_learning"})
        return findings

    def _check_threat_intel(self, command):
        findings = []
        now = time.time()
        if now - self.threat_intel_last_update > 7200:
            return findings
        with self._cache_lock:
            for adv in self.threat_intel_cache.get("github", []):
                cve = adv.get("ghsa_id", "")
                if cve.lower() in command.lower():
                    findings.append({"type": "threat_intel", "source": "github", "severity": "high", "description": f"Vulnerability: {cve}", "confidence": 0.9})
        return findings

    def _detect_anomalies(self, command, session_ctx):
        if not session_ctx.learning_enabled:
            return 0.0
        features = sum([len(command), command.count("|"), command.count(";"), command.count("&&"), len(command.split())]) / 5.0
        self.command_baselines[session_ctx.session_id].append(features)
        if len(self.command_baselines[session_ctx.session_id]) > 100:
            self.command_baselines[session_ctx.session_id] = self.command_baselines[session_ctx.session_id][-100:]
        if len(self.command_baselines[session_ctx.session_id]) < 10:
            return 0.0
        baseline = self.command_baselines[session_ctx.session_id][:-1]
        mean = sum(baseline) / len(baseline)
        var = sum((x - mean) ** 2 for x in baseline) / len(baseline)
        std = var ** 0.5
        return abs(features - mean) / std if std > 0 else 0.0

    def record_incident(self, incident_type, severity, description, context, user_feedback=None):
        incident_id = hashlib.sha256(f"{time.time()}{incident_type}".encode()).hexdigest()[:16]
        incident = SecurityIncident(id=incident_id, timestamp=time.time(), incident_type=incident_type, severity=severity, description=description, context=context, user_feedback=user_feedback)
        self.incidents[incident_id] = incident
        if user_feedback:
            self._learn_from_incident(incident)
        return incident_id

    def _learn_from_incident(self, incident):
        cmd = incident.context.get("command", "")
        if incident.user_feedback == "false_positive":
            self._reduce_pattern_confidence(cmd)
        elif incident.user_feedback == "blocked":
            self._create_blocking_pattern(cmd, incident.incident_type)

    def _reduce_pattern_confidence(self, command):
        with self._cache_lock:
            for p in self.patterns.values():
                if p.pattern_type == "substring" and p.pattern_data in command:
                    p.false_positive_count += 1
                    p.confidence = max(0.1, p.confidence * 0.8)

    def _create_blocking_pattern(self, command, incident_type):
        if incident_type == "path_traversal":
            for p in re.findall(r"\.\./+", command):
                pid = hashlib.sha256(p.encode()).hexdigest()[:12]
                if pid not in self.patterns:
                    self.patterns[pid] = ThreatPattern(pattern_id=pid, pattern_type="substring", pattern_data=p, confidence=0.7, source="learned")

    def get_security_report(self, session_id=None):
        incidents = list(self.incidents.values())
        return {"total_incidents": len(incidents), "learned_patterns": len([p for p in self.patterns.values() if p.source == "learned"]), "patterns": [{"id": p.pattern_id, "confidence": p.confidence} for p in sorted(self.patterns.values(), key=lambda x: x.confidence, reverse=True)[:5]]}

    def export_learning_data(self):
        return json.dumps({"version": "1.0", "patterns": [{"pattern_id": p.pattern_id, "pattern_type": p.pattern_type, "pattern_data": p.pattern_data, "confidence": p.confidence} for p in self.patterns.values() if p.source == "learned"]})

    def import_learning_data(self, data):
        try:
            imported = 0
            for pd in json.loads(data).get("patterns", []):
                pid = pd.get("pattern_id")
                if pid and pid not in self.patterns:
                    self.patterns[pid] = ThreatPattern(pattern_id=pid, pattern_type=pd.get("pattern_type", "substring"), pattern_data=pd.get("pattern_data", ""), confidence=min(pd.get("confidence", 0.5), 0.8), source="imported")
                    imported += 1
            return {"success": True, "imported": imported}
        except Exception as e:
            return {"success": False, "error": str(e)}


_security_engine = None
_engine_lock = threading.Lock()

def get_security_engine():
    global _security_engine
    with _engine_lock:
        if _security_engine is None:
            _security_engine = SelfLearningSecurity()
        return _security_engine

def analyze_command(command, context=None):
    return get_security_engine().analyze_command(command, context)

def record_incident(incident_type, severity, description, context, user_feedback=None):
    return get_security_engine().record_incident(incident_type, severity, description, context, user_feedback)
