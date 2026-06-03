"""
fact_feedback_loop.py — 自动校准 Holographic 信任分反馈闭环

直接操作 memory_store.db 的 SQLite 层（绕过 tool 系统），
基于统计规则自动给事实加减信任分，形成反馈闭环。

使用方式：
  python3 -m agent.fact_feedback_loop           # 查看健康报告
  python3 -m agent.fact_feedback_loop --calibrate  # 执行校准（dry-run）
  python3 -m agent.fact_feedback_loop --calibrate --apply  # 实际校准

规则集（三层递进）：
  1) 高检索低反馈降权: 检索≥3次但零反馈 → trust -= 0.1（冷落惩罚）
  2) 高反馈比例升权:  反馈/检索 > 30%   → trust += 0.05（正反馈奖励）
  3) 零检索淘汰衰减:  创建>30天且零检索 → trust *= 0.95（遗忘曲线）
  4) 边界保护:        trust ∈ [0.05, 0.95]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).resolve().parent.parent / "memory_store_calib.db"
MIN_TRUST = 0.05
MAX_TRUST = 0.95

RETRIEVAL_WITHOUT_FEEDBACK_THRESHOLD = 3   # 检索 N 次无反馈就降权
STALE_DAYS = 14                             # 超过 N 天零检索视为过时
HIGH_FEEDBACK_RATIO = 0.3                  # 反馈/检索 > N 就升权


# ── DB helpers ─────────────────────────────────────────────────────────────

def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            fact_id TEXT PRIMARY KEY,
            content TEXT,
            statement TEXT,
            confidence REAL,
            score REAL,
            trust_score REAL,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            unhelpful_count INTEGER DEFAULT 0,
            status TEXT,
            source TEXT,
            valid INTEGER,
            created_at TEXT,
            updated_at TEXT,
            last_retrieved_at TEXT
        )
    """)
    return conn

def clamp_trust(t: float) -> float:
    """将信任分限制在 [MIN_TRUST, MAX_TRUST] 区间"""
    return max(MIN_TRUST, min(MAX_TRUST, t))


def now_iso() -> str:
    return datetime.now().isoformat()


# ── Analysis ───────────────────────────────────────────────────────────────

def analyze_health() -> dict[str, Any]:
    """生成 fact 库全局健康报告"""
    conn = get_db()

    stats = dict(conn.execute("""
        SELECT
            COUNT(*)                                                AS total,
            SUM(CASE WHEN retrieval_count = 0 THEN 1 ELSE 0 END)   AS never_retrieved,
            SUM(CASE WHEN retrieval_count > 0 AND helpful_count = 0
                      THEN 1 ELSE 0 END)                            AS retrieved_no_feedback,
            SUM(CASE WHEN helpful_count > 0 THEN 1 ELSE 0 END)     AS ever_helpful,
            ROUND(AVG(trust_score), 4)                              AS avg_trust,
            SUM(retrieval_count)                                    AS total_retrievals,
            SUM(helpful_count)                                      AS total_helpful,
            ROUND(AVG(CASE WHEN retrieval_count > 0
                      THEN CAST(helpful_count AS REAL) / retrieval_count
                      ELSE NULL END), 4)                            AS avg_feedback_ratio
        FROM facts
    """).fetchone())

    # 分档统计
    tiers = {}
    for row in conn.execute("""
        SELECT
            CASE
                WHEN trust_score >= 0.8 THEN 'high(≥0.8)'
                WHEN trust_score >= 0.5 THEN 'mid(0.5-0.8)'
                WHEN trust_score >= 0.2 THEN 'low(0.2-0.5)'
                ELSE 'dead(<0.2)'
            END AS tier,
            COUNT(*) AS cnt
        FROM facts
        GROUP BY tier
        ORDER BY tier
    """).fetchall():
        tiers[row["tier"]] = row["cnt"]
    stats["trust_tiers"] = tiers

    # 需校准的候选
    cutoff = (datetime.now() - timedelta(days=STALE_DAYS)).isoformat()

    candidates = []
    for row in conn.execute("""
        SELECT fact_id, content, trust_score, retrieval_count, helpful_count,
               created_at
        FROM facts
        WHERE retrieval_count >= ?
          AND helpful_count = 0
        ORDER BY retrieval_count DESC
        LIMIT 50
    """, (RETRIEVAL_WITHOUT_FEEDBACK_THRESHOLD,)).fetchall():
        candidates.append({
            "fact_id": row["fact_id"],
            "reason": "high_retrieval_no_feedback",
            "trust_score": row["trust_score"],
            "retrieval_count": row["retrieval_count"],
            "helpful_count": row["helpful_count"],
            "preview": row["content"][:60],
        })

    for row in conn.execute("""
        SELECT fact_id, content, trust_score, retrieval_count, helpful_count
        FROM facts
        WHERE retrieval_count > 0
          AND CAST(helpful_count AS REAL) / retrieval_count > ?
        ORDER BY CAST(helpful_count AS REAL) / retrieval_count DESC
        LIMIT 50
    """, (HIGH_FEEDBACK_RATIO,)).fetchall():
        ratio = row["helpful_count"] / row["retrieval_count"]
        candidates.append({
            "fact_id": row["fact_id"],
            "reason": "high_feedback_ratio",
            "trust_score": row["trust_score"],
            "retrieval_count": row["retrieval_count"],
            "helpful_count": row["helpful_count"],
            "ratio": round(ratio, 2),
            "preview": row["content"][:60],
        })

    count_zero_retrieval_stale = conn.execute("""
        SELECT COUNT(*) FROM facts
        WHERE retrieval_count = 0 AND created_at < ?
    """, (cutoff,)).fetchone()[0]
    stats["zero_retrieval_stale"] = count_zero_retrieval_stale

    stats["candidates"] = candidates
    stats["retrieval_distribution"] = {}
    for row in conn.execute("""
        SELECT retrieval_count, COUNT(*) as cnt
        FROM facts GROUP BY retrieval_count ORDER BY retrieval_count
    """).fetchall():
        stats["retrieval_distribution"][row["retrieval_count"]] = row["cnt"]

    conn.close()
    return stats


# ── Calibration ────────────────────────────────────────────────────────────

def calibrate(dry_run: bool = True) -> dict[str, Any]:
    """执行信任分批量化校准

    Args:
        dry_run: True=只报告不修改, False=实际写入DB

    Returns:
        { dry_run, total_actions, downgrades, upgrades, decays, actions: [...] }
    """
    conn = get_db()
    actions: list[dict[str, Any]] = []
    now_str = now_iso()

    # ── 规则 1：高检索 + 零反馈 → 降权 ──
    for row in conn.execute("""
        SELECT fact_id, content, trust_score, retrieval_count, helpful_count
        FROM facts
        WHERE retrieval_count >= ?
          AND helpful_count = 0
    """, (RETRIEVAL_WITHOUT_FEEDBACK_THRESHOLD,)).fetchall():
        new_trust = clamp_trust(row["trust_score"] - 0.1)
        if new_trust != row["trust_score"]:
            if not dry_run:
                conn.execute("UPDATE facts SET trust_score=?, updated_at=? WHERE fact_id=?",
                            (new_trust, now_str, row["fact_id"]))
            actions.append({
                "fact_id": row["fact_id"], "action": "downgrade",
                "from": row["trust_score"], "to": new_trust,
                "reason": f"retrieved {row['retrieval_count']}x, 0 feedback",
                "preview": row["content"][:60],
            })

    # ── 规则 2：高反馈比例 → 升权 ──
    for row in conn.execute("""
        SELECT fact_id, content, trust_score, retrieval_count, helpful_count
        FROM facts
        WHERE retrieval_count > 0
          AND CAST(helpful_count AS REAL) / retrieval_count > ?
    """, (HIGH_FEEDBACK_RATIO,)).fetchall():
        new_trust = clamp_trust(row["trust_score"] + 0.05)
        if new_trust != row["trust_score"]:
            if not dry_run:
                conn.execute("UPDATE facts SET trust_score=?, updated_at=? WHERE fact_id=?",
                            (new_trust, now_str, row["fact_id"]))
            ratio = row["helpful_count"] / row["retrieval_count"]
            actions.append({
                "fact_id": row["fact_id"], "action": "upgrade",
                "from": row["trust_score"], "to": new_trust,
                "reason": f"helpful/retrieval={row['helpful_count']}/{row['retrieval_count']}={ratio:.0%}",
                "preview": row["content"][:60],
            })

    # ── 规则 3：零检索超过 STALE_DAYS → 衰减 ──
    cutoff = (datetime.now() - timedelta(days=STALE_DAYS)).isoformat()
    for row in conn.execute("""
        SELECT fact_id, content, trust_score, retrieval_count,
               created_at
        FROM facts
        WHERE retrieval_count = 0
          AND created_at < ?
          AND trust_score > ?
        ORDER BY created_at ASC
        LIMIT 500
    """, (cutoff, MIN_TRUST + 0.01)).fetchall():
        new_trust = clamp_trust(row["trust_score"] * 0.95)
        if new_trust != row["trust_score"]:
            if not dry_run:
                conn.execute("UPDATE facts SET trust_score=?, updated_at=? WHERE fact_id=?",
                            (new_trust, now_str, row["fact_id"]))
            actions.append({
                "fact_id": row["fact_id"], "action": "decay",
                "from": row["trust_score"], "to": new_trust,
                "reason": f"0 retrievals in {STALE_DAYS}d, created {row['created_at']}",
                "preview": row["content"][:60],
            })

    if not dry_run:
        conn.commit()

    conn.close()
    return {
        "dry_run": dry_run,
        "total_actions": len(actions),
        "downgrades": sum(1 for a in actions if a["action"] == "downgrade"),
        "upgrades": sum(1 for a in actions if a["action"] == "upgrade"),
        "decays": sum(1 for a in actions if a["action"] == "decay"),
        "actions": actions[:30],  # 只返回前 30 条详情
    }


# ── Report Formatter ──────────────────────────────────────────────────────

def format_health_report(stats: dict[str, Any]) -> str:
    """生成人类可读的健康报告"""
    lines = []
    lines.append("## Holographic 健康报告")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 总 facts | {stats['total']} |")
    lines.append(f"| 曾检索 | {stats['total_retrievals']} |")
    lines.append(f"| 曾反馈 | {stats['total_helpful']} |")
    lines.append(f"| 零检索 | {stats['never_retrieved']} ({stats['never_retrieved']/stats['total']*100:.0f}%) |")
    lines.append(f"| 零检索>30天 | {stats['zero_retrieval_stale']} |")
    lines.append(f"| 平均信任分 | {stats['avg_trust']:.3f} |")
    lines.append(f"| 平均反馈率 | {stats['avg_feedback_ratio'] or 'N/A'} |")
    lines.append("")

    lines.append("### 信任分分布")
    for tier, cnt in sorted(stats["trust_tiers"].items()):
        bar = "█" * max(1, cnt // 50)
        lines.append(f"  {tier}: {cnt:>5} {bar}")
    lines.append("")

    lines.append("### 检索次数分布")
    for k in sorted(stats["retrieval_distribution"].keys()):
        v = stats["retrieval_distribution"][k]
        label = f"retrieval={k}" if k <= 1 else f"retrieval≥{k}"
        bar = "█" * max(1, v // 100)
        lines.append(f"  {label:>15}: {v:>5} {bar}")
    lines.append("")

    if stats["candidates"]:
        lines.append("### 需校准的 fact")
        for c in stats["candidates"]:
            lines.append(f"  #{c['fact_id']} [{c['reason']}] trust={c['trust_score']:.2f} | {c['preview']}")
        lines.append("")
    else:
        lines.append("### 需校准的 fact")
        lines.append("  (无)")
        lines.append("")

    total_recallable = stats['total'] - stats['never_retrieved']
    lines.append(f"**总结**: {stats['total_retrievals']} 次检索集中在 {total_recallable} 条 fact 上，")
    lines.append(f"剩余 {stats['never_retrieved']} 条 ({stats['never_retrieved']/stats['total']*100:.0f}%) 从未被召回——")
    lines.append(f"建议执行 `--calibrate --apply` 以衰减陈旧低价值事实。")

    return "\n".join(lines)


def format_calibrate_report(result: dict[str, Any]) -> str:
    """生成校准报告"""
    lines = []
    mode = "[DRY RUN]" if result["dry_run"] else "[APPLIED]"
    lines.append(f"## 信任分校准报告 {mode}")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 降权 | {result['downgrades']} |")
    lines.append(f"| 升权 | {result['upgrades']} |")
    lines.append(f"| 衰减 | {result['decays']} |")
    lines.append(f"| 合计 | {result['total_actions']} |")
    lines.append("")

    if result["actions"]:
        lines.append("### 详情（前 30 条）")
        for a in result["actions"]:
            arrow = "↓" if a["action"] == "downgrade" else ("↑" if a["action"] == "upgrade" else "↘")
            lines.append(f"  #{a['fact_id']} {arrow} {a['from']:.2f}→{a['to']:.2f} [{a['reason']}] | {a['preview']}")
        lines.append("")

    if result["dry_run"]:
        lines.append("> 本次为 DRY RUN，未实际写入。加 --apply 执行。")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Holographic fact_feedback 闭环校准")
    parser.add_argument("--calibrate", action="store_true", help="执行校准")
    parser.add_argument("--apply", action="store_true", help="实际写入（默认 dry-run）")
    parser.add_argument("--report", action="store_true", help="只输出健康报告（默认）")
    args = parser.parse_args()

    if args.calibrate:
        dry_run = not args.apply
        result = calibrate(dry_run=dry_run)
        print(format_calibrate_report(result))
    else:
        stats = analyze_health()
        print(format_health_report(stats))


if __name__ == "__main__":
    main()
