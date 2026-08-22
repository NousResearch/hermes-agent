# skill-sleep — Hermes 技能自我进化

把 SkillOpt 的方法论（rollout → reflect → bounded edit → validation gate → staged adoption）用 Hermes 原生组件实现，**不依赖 Microsoft SkillOpt 包**。冻结 agent 不变，优化器在文本空间对 `SKILL.md` 做有界编辑，只有通过留出验证集的候选才被采纳。

> 设计参考：`hermes-agent#71266` + SkillOpt 论文 `arXiv:2605.23904`
> 详见 [DESIGN.md](DESIGN.md)

## 四阶段架构

```
skill-sleep run
  ├─ [1] MINE      扫近期会话（hermes sessions export --redact），提取 friction 信号 → tasks.json
  ├─ [2] PROPOSE   优化器（omp + gpt-4o-mini，默认模型可 --model 覆写）读 task cards + 当前 SKILL.md → candidate.diff（含 rejected buffer 上下文）
  ├─ [3] VALIDATE  LLM judge 逐 task 评分，阈值 + 通过率双 gate（默认 70 分 / 60%）→ validation.json
  └─ [4] REVIEW    PASS → staging/<skill>-<ts>/ 待人类审查 → skill_manage patch 采纳
                   FAIL → rejected/<skill>-<ts>/ + rejected.jsonl 进下一轮 prompt
```

## 快速开始

```bash
# 1. 准备环境（项目自带 .venv）
source .venv/bin/activate

# 2. 配置密钥（模型需在 omp 中可用；默认 gpt-4o-mini，可 --model 覆写）
export NINEROUTER_KEY=...

# 3. 单阶段手动运行（带 [stage] 前缀日志）
python3 pipeline/mine.py --after 7d --output-dir /tmp/run
python3 pipeline/propose.py --tasks /tmp/run/tasks.json --output-dir /tmp/run
python3 pipeline/validate.py --tasks /tmp/run/tasks.json --diff /tmp/run/candidate.diff --proposal /tmp/run/proposal.json --output-dir /tmp/run
python3 pipeline/review.py stage --validation /tmp/run/validation.json --diff /tmp/run/candidate.diff --proposal /tmp/run/proposal.json --output-dir .

# 本地联调可用 --dry-run 跳过 omp 调用（启发式占位）
python3 pipeline/propose.py --tasks /tmp/run/tasks.json --output-dir /tmp/run --dry-run
python3 pipeline/validate.py --tasks /tmp/run/tasks.json --diff /tmp/run/candidate.diff --proposal /tmp/run/proposal.json --output-dir /tmp/run --dry-run

# 4. 一键串联（cron 入口）
bash skill-sleep.sh
# 每步检查前置文件是否存在，缺则优雅跳过；自动检测 .venv 与 NINEROUTER_KEY

# 5. 审查与采纳
ls staging/                              # 查看待审查候选
python3 pipeline/review.py apply --staging-dir staging/<skill>-<ts>
python3 pipeline/review.py reject --staging-dir staging/<skill>-<ts> --reason "..."
```

## 目录

```
.
├── DESIGN.md                 设计文档
├── skill-sleep.sh            cron 入口（串联 MINE → PROPOSE → VALIDATE → REVIEW）
├── config.yaml               配置（可选）
├── pipeline/
│   ├── mine.py               会话扫描 + friction 提取
│   ├── propose.py            omp + muse 生成候选 diff
│   ├── validate.py           gate：逐 task 评分 + 聚合
│   └── review.py             staging / rejected / apply / reject
├── lib/                      TaskCard / ValidationResult / staging 工具
├── templates/                propose / judge 提示词模板
└── tests/                    pytest（134 tests PASS）
```

## 配置与阈值

- `PROPOSE`: `DEFAULT_MODEL=gpt-4o-mini`，diff 有界（≤30 行新增，文本学习率）
- `VALIDATE`: `threshold=70`，`min_pass_rate=0.6`，`gate_type=llm_judge`
- 隐私：`hermes sessions export --redact`，task card 脱敏后再送优化器；muse spark 有数据分享条款

## 局限（诚实）

- Replay fidelity 有限：有副作用的 skill 不能安全重放两次，验证用 LLM judge 而非确定性复现
- 非确定性：LLM 评分方差大，建议多轮取均值或接受方差
- 见 DESIGN.md「与 SkillOpt 的区别」

## 测试

```bash
.venv/bin/python -m pytest -q
```
