"""app_skill_reload_simulator —— 模拟 Skill 变更后的缓存失效与重新加载流程。

依据飞书文档《Hermes-Agent Skill 加载机制》第三、四章设计：
    路径 A: System Prompt 索引 (LRU + Snapshot)
    路径 C: Slash Command 路由表 (_skill_commands)
    路径 D: Skill Bundle (_bundles_cache + _max_mtime)

本脚本会：
  1. 在临时目录创建 3 个最小化 skill (含 SKILL.md frontmatter)；
  2. 通过 monkeypatch 让 skill 子系统把临时目录当作 SKILLS_DIR；
  3. 演练 4 种变更：新增、修改 description、删除、Bundle 变更；
  4. 每次变更后打印 "变更前/变更后" 缓存状态，验证失效路径正确。

运行：
    python app_skill_reload_simulator.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Dict


# --------------------------------------------------------------------------- #
# Step 0. 准备临时 skills 目录与 bundles 目录，并切到对应 HERMES_HOME
# --------------------------------------------------------------------------- #
SANDBOX = Path(tempfile.mkdtemp(prefix="hermes-skill-sim-"))
SKILLS_DIR = SANDBOX / "skills"
BUNDLES_DIR = SANDBOX / "skill-bundles"
SKILLS_DIR.mkdir(parents=True, exist_ok=True)
BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

# 必须在 import hermes_constants 之前设置，让 get_hermes_home 指向沙箱
os.environ["HERMES_HOME"] = str(SANDBOX)

# 现在再导入项目模块（它们会读取 HERMES_HOME）
sys.path.insert(0, str(Path(__file__).parent))
from agent import skill_bundles, skill_commands  # noqa: E402
from agent.prompt_builder import (  # noqa: E402
    _SKILLS_PROMPT_CACHE,
    _build_skills_manifest,
    clear_skills_system_prompt_cache,
)
from tools import skills_tool  # noqa: E402

# 把模块内硬编码的 SKILLS_DIR / BUNDLES_DIR 指向沙箱
skills_tool.SKILLS_DIR = SKILLS_DIR
skill_bundles.BUNDLES_DIR = BUNDLES_DIR


# --------------------------------------------------------------------------- #
# 工具函数：写一个最小化的 skill
# --------------------------------------------------------------------------- #
def write_skill(name: str, description: str) -> Path:
    skill_dir = SKILLS_DIR / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    md = skill_dir / "SKILL.md"
    md.write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: {description}
            ---
            # {name}

            模拟 skill 主体内容。
            """
        ),
        encoding="utf-8",
    )
    # 让 mtime 至少递增 1 秒，避免缓存命中假阴性
    time.sleep(0.01)
    return md


def write_bundle(slug: str, skills: list[str]) -> Path:
    path = BUNDLES_DIR / f"{slug}.yaml"
    body = "name: {slug}\ndescription: bundle of {slug}\nskills:\n".format(slug=slug)
    body += "\n".join(f"  - {s}" for s in skills) + "\n"
    path.write_text(body, encoding="utf-8")
    time.sleep(0.01)
    return path


def banner(msg: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {msg}")
    print("=" * 72)


def show_state(tag: str) -> None:
    cmds = list(skill_commands._skill_commands.keys())
    bundles = list(skill_bundles._bundles_cache.keys())
    prompt_cache_size = len(_SKILLS_PROMPT_CACHE)
    print(f"[{tag}]")
    print(f"  /commands       = {sorted(cmds)}")
    print(f"  bundles         = {sorted(bundles)}")
    print(f"  prompt LRU size = {prompt_cache_size}")


# --------------------------------------------------------------------------- #
# Step 1. 初始装载 (冷启动)
# --------------------------------------------------------------------------- #
def step_initial_load() -> None:
    banner("Step 1. 初始装载：写入 3 个 skill + 1 个 bundle，触发首次扫描")
    write_skill("app-foo", "Foo skill: 处理订单聚合任务")
    write_skill("app-bar", "Bar skill: 拉取多维表格")
    write_skill("app-baz", "Baz skill: 推送告警通知")
    write_bundle("app-pipeline", ["app-foo", "app-bar"])

    # 触发路径 C 的首次 scan
    skill_commands.scan_skill_commands()
    # 触发路径 D 的首次 scan
    skill_bundles.scan_bundles()
    # 模拟路径 A 的 system prompt 缓存填充：直接放入一项，演示失效
    manifest = _build_skills_manifest(SKILLS_DIR)
    manifest_key = tuple(sorted((k, tuple(v)) for k, v in manifest.items()))
    _SKILLS_PROMPT_CACHE[("k1", manifest_key)] = "<dummy prompt>"

    show_state("initial")


# --------------------------------------------------------------------------- #
# Step 2. 变更类型 A：新增 skill
# --------------------------------------------------------------------------- #
def step_add_skill() -> None:
    banner("Step 2. 变更：新增 skill `app-qux`")
    show_state("before-add")

    write_skill("app-qux", "Qux skill: 用户画像同步")

    # 失效与重扫（模拟 cli._reload_skills 的最小动作）
    diff = skill_commands.reload_skills()
    print(f"  reload_skills() diff = added={[d['name'] for d in diff['added']]}, "
          f"removed={[d['name'] for d in diff['removed']]}")

    # System Prompt 缓存随 manifest 变化而天然失效；为演示，主动清一次
    clear_skills_system_prompt_cache()
    show_state("after-add")


# --------------------------------------------------------------------------- #
# Step 3. 变更类型 B：修改 description（覆盖写）
# --------------------------------------------------------------------------- #
def step_modify_skill() -> None:
    banner("Step 3. 变更：修改 `app-bar` 的 description")
    show_state("before-modify")

    write_skill("app-bar", "Bar skill v2: 升级后的多维表格读写器")

    # description 变化只能通过重扫感知；reload_skills 内部会 reset 字典并重写
    diff = skill_commands.reload_skills()
    new_desc = skill_commands._skill_commands.get("/app-bar", {}).get("description")
    print(f"  /app-bar 新 description = {new_desc!r}")
    print(f"  diff.unchanged 包含 app-bar? "
          f"{'app-bar' in diff['unchanged']}  # 名称未变，仅描述更新")

    # System Prompt 缓存：manifest 中 mtime 变了，下次 build 会 miss 进而重建
    clear_skills_system_prompt_cache()
    show_state("after-modify")


# --------------------------------------------------------------------------- #
# Step 4. 变更类型 C：删除 skill
# --------------------------------------------------------------------------- #
def step_remove_skill() -> None:
    banner("Step 4. 变更：删除 `app-baz`")
    show_state("before-remove")

    shutil.rmtree(SKILLS_DIR / "app-baz")

    diff = skill_commands.reload_skills()
    print(f"  reload_skills() diff.removed = {[d['name'] for d in diff['removed']]}")

    clear_skills_system_prompt_cache()
    show_state("after-remove")


# --------------------------------------------------------------------------- #
# Step 5. 变更类型 D：Bundle 变更（基于 _max_mtime 自动失效）
# --------------------------------------------------------------------------- #
def step_modify_bundle() -> None:
    banner("Step 5. 变更：更新 `app-pipeline` bundle，引用最新 skill 集合")
    show_state("before-bundle-update")

    # 扩展 bundle 引用列表
    write_bundle("app-pipeline", ["app-foo", "app-bar", "app-qux"])

    # 不需要主动 reload：get_skill_bundles 会比较 _max_mtime，自动重扫
    bundles = skill_bundles.get_skill_bundles()
    pipeline = bundles.get("/app-pipeline") or bundles.get("app-pipeline") or {}
    print(f"  /app-pipeline.skills = {pipeline.get('skills')}")

    show_state("after-bundle-update")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"沙箱目录: {SANDBOX}")
    try:
        step_initial_load()
        step_add_skill()
        step_modify_skill()
        step_remove_skill()
        step_modify_bundle()
        banner("Done. 全部场景演练完成。")
    finally:
        # 清理沙箱（如需保留产物可注释掉）
        shutil.rmtree(SANDBOX, ignore_errors=True)


if __name__ == "__main__":
    main()
