"""
EvoMaster - CLAW 原生进化核心引擎

实现 CLAW 迭代进化核心公式：
1. 迭代进化总目标：max E[R_exec(τ) + λ·K_claw(τ)]
2. 策略自更新：π^(t+1) = GPT-Stream(τ^(t), K_claw, Constraint)
3. 知识缓存压缩：K_claw = HashPool(Filter(τ_valid))
"""

import json
import hashlib
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trajectory:
    """执行轨迹"""
    id: str
    task: str
    actions: List[Dict[str, Any]]
    success: bool
    reward: float  # R_exec: 执行成功率收益
    knowledge_value: float  # K_claw: 知识沉淀价值
    total_value: float  # R_exec + λ·K_claw
    created_at: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def fingerprint(self) -> str:
        """轨迹指纹（用于去重和缓存）"""
        content = json.dumps({
            'task': self.task,
            'actions': self.actions,
            'success': self.success
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Strategy:
    """CLAW 调度策略"""
    id: str
    version: int
    policy: Dict[str, Any]  # π_claw: 策略参数
    performance: float  # 性能指标
    created_at: str
    metadata: Dict[str, Any] = None


class KnowledgeCache:
    """知识缓存池（K_claw）"""
    
    def __init__(self, db_path: str):
        """
        初始化知识缓存
        
        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 轨迹表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id TEXT PRIMARY KEY,
                fingerprint TEXT UNIQUE,
                task TEXT,
                actions TEXT,
                success INTEGER,
                reward REAL,
                knowledge_value REAL,
                total_value REAL,
                created_at TEXT,
                metadata TEXT
            )
        """)
        
        # 策略表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id TEXT PRIMARY KEY,
                version INTEGER,
                policy TEXT,
                performance REAL,
                created_at TEXT,
                metadata TEXT
            )
        """)
        
        # 索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fingerprint ON trajectories(fingerprint)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_total_value ON trajectories(total_value DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy_version ON strategies(version DESC)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ 知识缓存初始化完成: {self.db_path}")
    
    def add_trajectory(self, trajectory: Trajectory) -> bool:
        """
        添加轨迹到缓存
        
        Args:
            trajectory: 轨迹对象
            
        Returns:
            是否成功（去重）
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查指纹是否已存在
            cursor.execute("""
                SELECT id FROM trajectories WHERE fingerprint = ?
            """, (trajectory.fingerprint,))
            
            if cursor.fetchone():
                conn.close()
                logger.debug(f"⏭️ 轨迹已存在（指纹: {trajectory.fingerprint}）")
                return False
            
            # 插入新轨迹
            cursor.execute("""
                INSERT INTO trajectories (
                    id, fingerprint, task, actions, success,
                    reward, knowledge_value, total_value,
                    created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trajectory.id,
                trajectory.fingerprint,
                trajectory.task,
                json.dumps(trajectory.actions),
                1 if trajectory.success else 0,
                trajectory.reward,
                trajectory.knowledge_value,
                trajectory.total_value,
                trajectory.created_at,
                json.dumps(trajectory.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ 添加轨迹: {trajectory.id} (价值: {trajectory.total_value:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 添加轨迹失败: {e}")
            return False
    
    def get_top_trajectories(self, limit: int = 10) -> List[Trajectory]:
        """
        获取最优轨迹
        
        Args:
            limit: 返回数量
            
        Returns:
            轨迹列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM trajectories
            WHERE success = 1
            ORDER BY total_value DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_trajectory(row) for row in rows]
    
    def get_similar_trajectories(self, task: str, limit: int = 5) -> List[Trajectory]:
        """
        获取相似任务的轨迹
        
        Args:
            task: 任务描述
            limit: 返回数量
            
        Returns:
            轨迹列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 简单的关键词匹配
        cursor.execute("""
            SELECT * FROM trajectories
            WHERE task LIKE ? AND success = 1
            ORDER BY total_value DESC
            LIMIT ?
        """, (f"%{task}%", limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_trajectory(row) for row in rows]
    
    def add_strategy(self, strategy: Strategy) -> bool:
        """添加策略"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO strategies (
                    id, version, policy, performance, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                strategy.id,
                strategy.version,
                json.dumps(strategy.policy),
                strategy.performance,
                strategy.created_at,
                json.dumps(strategy.metadata or {})
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ 添加策略: v{strategy.version} (性能: {strategy.performance:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 添加策略失败: {e}")
            return False
    
    def get_latest_strategy(self) -> Optional[Strategy]:
        """获取最新策略"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM strategies
            ORDER BY version DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Strategy(
                id=row[0],
                version=row[1],
                policy=json.loads(row[2]),
                performance=row[3],
                created_at=row[4],
                metadata=json.loads(row[5]) if row[5] else {}
            )
        return None
    
    def _row_to_trajectory(self, row) -> Trajectory:
        """将数据库行转换为 Trajectory 对象"""
        return Trajectory(
            id=row[0],
            task=row[2],
            actions=json.loads(row[3]),
            success=bool(row[4]),
            reward=row[5],
            knowledge_value=row[6],
            total_value=row[7],
            created_at=row[8],
            metadata=json.loads(row[9]) if row[9] else {}
        )


class EvoMaster:
    """进化主控引擎"""
    
    def __init__(
        self,
        knowledge_cache: KnowledgeCache,
        lambda_weight: float = 0.3
    ):
        """
        初始化进化引擎
        
        Args:
            knowledge_cache: 知识缓存
            lambda_weight: 知识复用权重 λ
        """
        self.knowledge_cache = knowledge_cache
        self.lambda_weight = lambda_weight
        self.current_strategy = knowledge_cache.get_latest_strategy()
        
        if not self.current_strategy:
            # 初始化默认策略
            self.current_strategy = Strategy(
                id="strategy_v0",
                version=0,
                policy={'type': 'default'},
                performance=0.0,
                created_at=datetime.utcnow().isoformat()
            )
            knowledge_cache.add_strategy(self.current_strategy)
    
    def record_trajectory(
        self,
        task: str,
        actions: List[Dict[str, Any]],
        success: bool,
        reward: float
    ) -> Trajectory:
        """
        记录执行轨迹
        
        Args:
            task: 任务描述
            actions: 动作序列
            success: 是否成功
            reward: 执行收益 R_exec
            
        Returns:
            轨迹对象
        """
        # 计算知识价值 K_claw
        knowledge_value = self._calculate_knowledge_value(task, actions, success)
        
        # 计算总价值：R_exec + λ·K_claw
        total_value = reward + self.lambda_weight * knowledge_value
        
        trajectory = Trajectory(
            id=f"traj_{datetime.utcnow().timestamp()}",
            task=task,
            actions=actions,
            success=success,
            reward=reward,
            knowledge_value=knowledge_value,
            total_value=total_value,
            created_at=datetime.utcnow().isoformat()
        )
        
        # 添加到知识缓存
        self.knowledge_cache.add_trajectory(trajectory)
        
        return trajectory
    
    def evolve_strategy(self, use_llm: bool = True) -> Strategy:
        """
        策略自更新迭代
        
        实现公式：π^(t+1) = GPT-Stream(τ^(t), K_claw, Constraint)
        
        Args:
            use_llm: 是否使用 LLM 流式推理生成策略（默认 True）
        
        Returns:
            新策略
        """
        # 获取最优轨迹
        top_trajectories = self.knowledge_cache.get_top_trajectories(limit=10)
        
        if not top_trajectories:
            logger.warning("⚠️ 无可用轨迹，保持当前策略")
            return self.current_strategy
        
        # 计算新策略性能
        avg_performance = sum(t.total_value for t in top_trajectories) / len(top_trajectories)
        
        if use_llm:
            # LLM 流式推理生成策略
            new_policy = self._llm_generate_policy(top_trajectories)
        else:
            # 简化版本（统计方法）
            new_policy = {
                'type': 'evolved',
                'base_version': self.current_strategy.version,
                'top_patterns': [
                    {
                        'task': t.task,
                        'actions': t.actions,
                        'value': t.total_value
                    }
                    for t in top_trajectories[:3]
                ]
            }
        
        new_strategy = Strategy(
            id=f"strategy_v{self.current_strategy.version + 1}",
            version=self.current_strategy.version + 1,
            policy=new_policy,
            performance=avg_performance,
            created_at=datetime.utcnow().isoformat()
        )
        
        # 保存新策略
        self.knowledge_cache.add_strategy(new_strategy)
        self.current_strategy = new_strategy
        
        mode = "LLM" if use_llm else "统计"
        logger.info(f"🔄 策略进化: v{new_strategy.version} (性能: {avg_performance:.2f}, 模式: {mode})")
        
        return new_strategy
    
    def _llm_generate_policy(self, top_trajectories: List[Trajectory]) -> Dict[str, Any]:
        """
        使用 LLM 流式推理生成新策略。

        实现 GPT-Stream(τ^(t), K_claw, Constraint_sandbox)：先按 Hermes
        当前配置调用主模型；429/503/超时/解析失败时，真实调用本机量子
        路由选择健康模型降级重试；只有主通路和动态降级都失败，才明确
        标注统计 fallback，避免把统计简化冒充 LLM 进化。
        """
        errors: List[str] = []

        def _statistical_fallback(reason: str) -> Dict[str, Any]:
            return {
                'type': 'evolved_fallback',
                'base_version': self.current_strategy.version,
                'fallback_reason': reason,
                'llm_errors': errors[-3:],
                'top_patterns': [
                    {
                        'task': t.task,
                        'actions': t.actions,
                        'value': t.total_value
                    }
                    for t in top_trajectories[:3]
                ],
            }

        try:
            import os
            import re
            import yaml
            from openai import OpenAI
            from pathlib import Path
            try:
                from dotenv import load_dotenv
            except Exception:  # pragma: no cover - optional dependency guard
                load_dotenv = None

            hermes_home = Path(os.environ.get('HERMES_HOME') or '/Users/appleoppa/.hermes')
            if load_dotenv is not None:
                for env_path in (hermes_home / '.env', Path('/Users/appleoppa/.hermes/.env')):
                    if env_path.exists():
                        load_dotenv(str(env_path), override=False)

            config_path = hermes_home / 'config.yaml'
            if not config_path.exists():
                config_path = Path('/Users/appleoppa/.hermes/config.yaml')
            if not config_path.exists():
                raise FileNotFoundError(f"Hermes config not found: {config_path}")

            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            providers = config.get('providers', {}) or {}
            custom_providers_raw = config.get('custom_providers', {}) or {}
            custom_providers = {}
            if isinstance(custom_providers_raw, list):
                for cp in custom_providers_raw:
                    if isinstance(cp, dict) and cp.get('name'):
                        custom_providers[cp['name']] = cp
            elif isinstance(custom_providers_raw, dict):
                custom_providers = custom_providers_raw
            all_providers = {**providers, **custom_providers}
            if not all_providers:
                raise ValueError('No providers configured in config.yaml')

            def _strip_custom_prefix(provider: str) -> str:
                provider = (provider or '').strip()
                return provider.split(':', 1)[1] if provider.startswith('custom:') else provider

            def _provider_candidates() -> List[str]:
                main_provider = _strip_custom_prefix(
                    str(config.get('model', {}).get('provider') or config.get('main_provider') or config.get('default_provider') or '')
                )
                ordered = [main_provider]
                try:
                    from tools.model_failover_tool import auto_failover
                    model, provider = auto_failover(
                        task='EvoMaster strategy evolution LLM policy generation',
                        failed_model='',
                        failed_provider='',
                    )
                    if provider:
                        ordered.append(_strip_custom_prefix(provider))
                except Exception as exc:
                    errors.append(f'qr_route_unavailable: {type(exc).__name__}: {exc}')
                ordered.extend([
                    'deepseek_v4_flash',
                    'claude_opus47_5yuantoken',
                    'gpt55_5yuantoken',
                    'appleoppa',
                ])
                seen = set()
                result = []
                for provider in ordered:
                    if provider and provider in all_providers and provider not in seen:
                        seen.add(provider)
                        result.append(provider)
                return result

            trajectory_summary = "\n".join([
                f"- 任务: {t.task}, 成功: {t.success}, 价值: {t.total_value:.2f}, 动作数: {len(t.actions)}"
                for t in top_trajectories[:8]
            ])
            action_samples = json.dumps([
                {
                    'task': t.task,
                    'actions': t.actions[:6],
                    'reward': t.reward,
                    'knowledge_value': t.knowledge_value,
                    'total_value': t.total_value,
                    'metadata_keys': sorted((t.metadata or {}).keys())[:10],
                }
                for t in top_trajectories[:5]
            ], ensure_ascii=False)[:6000]
            prompt = f"""你是 CLAW/APEX 进化策略生成器。请基于真实执行轨迹生成下一版调度策略。

当前策略版本: v{self.current_strategy.version}
当前策略性能: {self.current_strategy.performance:.2f}

最优轨迹摘要:
{trajectory_summary}

轨迹样本JSON:
{action_samples}

约束条件:
- 必须保持工具调用安全、可逆、可验证。
- 优先复用成功率高、知识沉淀价值高的动作模式。
- 明确避免失败或低价值动作序列。
- 输出必须是严格 JSON，不要 Markdown，不要解释。

JSON schema:
{{
  "type": "llm_evolved",
  "key_insights": ["..."],
  "recommended_patterns": [{{"task_type": "...", "action_sequence": ["..."], "priority": 0.9, "evidence": "..."}}],
  "avoid_patterns": ["..."],
  "verification_gates": ["..."],
  "performance_target": 0.0
}}"""

            last_error = None
            for provider_name in _provider_candidates():
                provider_config = all_providers[provider_name]
                base_url = provider_config.get('base_url') or provider_config.get('api_base')
                key_env = provider_config.get('key_env') or provider_config.get('api_key_env')
                api_key = os.environ.get(key_env) if key_env else None
                api_key = api_key or provider_config.get('api_key')
                model_name = provider_config.get('default_model') or provider_config.get('model') or provider_config.get('name')
                if not base_url or not api_key or not model_name:
                    errors.append(f'{provider_name}: missing base_url/api_key/model')
                    continue
                try:
                    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
                    logger.info(f"🤖 调用 LLM 生成策略: {provider_name}/{model_name}")
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "你是 CLAW/APEX 进化策略专家，只输出严格 JSON。"},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=1200,
                    )
                    content = response.choices[0].message.content or ""
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if not json_match:
                        raise ValueError('No JSON in LLM response')
                    policy = json.loads(json_match.group())
                    if not isinstance(policy, dict):
                        raise ValueError('LLM JSON is not an object')
                    policy['type'] = policy.get('type') or 'llm_evolved'
                    policy['base_version'] = self.current_strategy.version
                    policy['llm_provider'] = provider_name
                    policy['llm_model'] = model_name
                    policy['llm_failover_errors'] = errors[-3:]
                    # 保留摘要和 hash，不保存完整原文，避免污染知识缓存。
                    policy['llm_response_sha256'] = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
                    logger.info(f"✅ LLM 生成策略: provider={provider_name}, insights={len(policy.get('key_insights', []))}")
                    return policy
                except Exception as exc:
                    last_error = exc
                    errors.append(f'{provider_name}: {type(exc).__name__}: {str(exc)[:180]}')
                    logger.warning("⚠️ LLM 策略生成失败，尝试下一通道: %s/%s error=%s", provider_name, model_name, exc)
                    continue

            return _statistical_fallback(f'all_llm_routes_failed: {last_error}')
        except Exception as e:
            errors.append(f'fatal: {type(e).__name__}: {str(e)[:180]}')
            logger.warning(f"⚠️ LLM 策略生成失败: {e}，降级到统计方法")
            return _statistical_fallback(str(e))

    def select_best_strategy_for_task(self, task: str) -> Optional[Strategy]:
        """
        根据任务类型选择最优策略版本。

        策略选择逻辑：
        1. 查找相似任务的成功轨迹（按 total_value 排序）
        2. 计算这些轨迹的平均性能
        3. 如果平均性能高于当前策略，触发策略进化

        Args:
            task: 任务描述

        Returns:
            最优策略
        """
        if self.current_strategy is None:
            return None
        
        # 查找相似任务的高价值轨迹
        similar_trajectories = self.knowledge_cache.get_similar_trajectories(task, limit=10)
        
        if not similar_trajectories:
            logger.debug(f"💡 无相似轨迹，使用当前策略 v{self.current_strategy.version}")
            return self.current_strategy
        
        # 计算相似任务的平均性能
        avg_performance = sum(t.total_value for t in similar_trajectories) / len(similar_trajectories)
        
        logger.info(
            f"💡 任务 '{task}' 选择策略 v{self.current_strategy.version} "
            f"(相似轨迹平均性能={avg_performance:.2f}, 数量={len(similar_trajectories)})"
        )
        return self.current_strategy
    
    def get_recommended_actions(self, task: str) -> List[Dict[str, Any]]:
        """
        基于知识缓存推荐动作序列
        
        Args:
            task: 任务描述
            
        Returns:
            推荐的动作序列
        """
        # 查找相似任务的成功轨迹
        similar_trajectories = self.knowledge_cache.get_similar_trajectories(task, limit=3)
        
        if not similar_trajectories:
            logger.debug(f"💡 无相似轨迹，使用默认策略")
            return []
        
        # 返回最优轨迹的动作序列
        best_trajectory = similar_trajectories[0]
        logger.info(f"💡 推荐动作（基于轨迹 {best_trajectory.id}，价值 {best_trajectory.total_value:.2f}）")
        
        return best_trajectory.actions
    
    def _calculate_knowledge_value(
        self,
        task: str,
        actions: List[Dict[str, Any]],
        success: bool
    ) -> float:
        """
        计算知识价值 K_claw
        
        考虑因素：
        1. 是否成功
        2. 动作序列复杂度
        3. 任务新颖性
        
        Args:
            task: 任务描述
            actions: 动作序列
            success: 是否成功
            
        Returns:
            知识价值（0-1）
        """
        if not success:
            return 0.0
        
        # 基础价值
        value = 0.5
        
        # 动作序列复杂度加分
        if len(actions) > 3:
            value += 0.2
        
        # 检查任务新颖性
        similar = self.knowledge_cache.get_similar_trajectories(task, limit=1)
        if not similar:
            # 新任务，价值更高
            value += 0.3
        
        return min(value, 1.0)

    def score_trajectories_with_eval_center(
        self,
        eval_center_db: str = '/Users/appleoppa/.hermes/eval_center.db',
    ) -> Dict[str, int]:
        """
        使用 Rust 评估中心对知识缓存中的轨迹进行评分。

        为已导入但未评分的轨迹调用 hermes_eval_center.score_trace()，
        更新 reward 和 total_value。

        Args:
            eval_center_db: 评估中心 SQLite 路径

        Returns:
            统计字典 {scored, skipped, total}
        """
        try:
            import hermes_eval_center
            ec = hermes_eval_center.PyEvalCenter(eval_center_db)
        except Exception as e:
            logger.warning(f"⚠️ 无法加载评估中心: {e}")
            return {'scored': 0, 'skipped': 0, 'total': 0}

        # 获取所有来自 eval_center 的轨迹
        conn = sqlite3.connect(self.knowledge_cache.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        rows = cur.execute("""
            SELECT id, metadata FROM trajectories
            WHERE metadata LIKE '%eval_center%'
        """).fetchall()
        
        scored = skipped = 0
        for row in rows:
            try:
                metadata = json.loads(row['metadata'])
                eval_center_id = metadata.get('eval_center_id')
                if not eval_center_id:
                    skipped += 1
                    continue

                # 调用评估中心评分
                score = ec.score_trace(eval_center_id)
                
                # 更新轨迹的 reward 和 total_value
                cur.execute("""
                    SELECT knowledge_value FROM trajectories WHERE id = ?
                """, (row['id'],))
                k_value = cur.fetchone()[0]
                
                new_total = score + self.lambda_weight * k_value
                
                cur.execute("""
                    UPDATE trajectories
                    SET reward = ?, total_value = ?
                    WHERE id = ?
                """, (score, new_total, row['id']))
                
                scored += 1
                
            except Exception as e:
                logger.debug(f"评分失败 {row['id']}: {e}")
                skipped += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ 评分完成: {scored}/{len(rows)} 条轨迹")
        return {'scored': scored, 'skipped': skipped, 'total': len(rows)}

    def import_from_eval_center(
        self,
        eval_center_db: str = '/Users/appleoppa/.hermes/eval_center.db',
        min_score: float = 0.0,
        only_active: bool = False,
        limit: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        从 Rust 评估中心 (hermes_eval_center) 批量导入真实轨迹。

        实现公式：K_claw = HashPool(Filter(τ_valid))
        其中 τ_valid 来自核心改造产生的真实工具调用轨迹。

        Args:
            eval_center_db: 评估中心 SQLite 路径
            min_score: 最低评分阈值
            only_active: 仅导入 state='active' 的轨迹
            limit: 最大导入数量

        Returns:
            统计字典 {imported, skipped, duplicate, total}
        """
        import sqlite3 as _sqlite

        if not Path(eval_center_db).exists():
            logger.warning(f"⚠️ 评估中心数据库不存在: {eval_center_db}")
            return {'imported': 0, 'skipped': 0, 'duplicate': 0, 'total': 0}

        conn = _sqlite.connect(eval_center_db)
        conn.row_factory = _sqlite.Row
        cur = conn.cursor()

        sql = "SELECT * FROM traces WHERE 1=1"
        params = []
        if only_active:
            sql += " AND state = 'active'"
        if min_score > 0:
            sql += " AND (score IS NULL OR score >= ?)"
            params.append(min_score)
        sql += " ORDER BY created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"

        rows = cur.execute(sql, params).fetchall()
        conn.close()

        imported = skipped = duplicate = 0
        for row in rows:
            try:
                tool_calls = json.loads(row['tool_calls']) if row['tool_calls'] else []
            except Exception:
                tool_calls = []

            # 推断成功状态
            score = row['score']
            state = row['state']
            tc_total = sum(1 for tc in tool_calls if isinstance(tc, dict) and 'success' in tc)
            tc_ok = sum(1 for tc in tool_calls if isinstance(tc, dict) and tc.get('success'))
            success = (
                state == 'active'
                or (score is not None and score >= 0.7)
                or (tc_total > 0 and tc_ok == tc_total)
            )

            # 估算 R_exec：评分 / 工具调用成功率
            if score is not None:
                reward = float(score)
            elif tc_total > 0:
                reward = tc_ok / tc_total
            else:
                reward = 0.5 if success else 0.0

            # 计算知识价值
            knowledge_value = self._calculate_knowledge_value(row['task'], tool_calls, success)
            total_value = reward + self.lambda_weight * knowledge_value

            traj = Trajectory(
                id=f"eval_center_{row['id'][:12]}",
                task=row['task'],
                actions=tool_calls if isinstance(tool_calls, list) else [],
                success=success,
                reward=reward,
                knowledge_value=knowledge_value,
                total_value=total_value,
                created_at=row['created_at'],
                metadata={
                    'source': 'eval_center',
                    'eval_center_id': row['id'],
                    'eval_center_state': state,
                    'eval_center_score': score,
                },
            )

            # 检查指纹去重并添加
            added = self.knowledge_cache.add_trajectory(traj)
            if not added:
                duplicate += 1
            else:
                imported += 1

        logger.info(
            f"✅ 从评估中心导入 {imported}/{len(rows)} 条轨迹 "
            f"(重复 {duplicate}, 跳过 {skipped})"
        )
        return {
            'imported': imported,
            'skipped': skipped,
            'duplicate': duplicate,
            'total': len(rows),
        }


# 便捷函数
def get_evo_master(db_path: Optional[str] = None, lambda_weight: float = 0.3) -> EvoMaster:
    """获取进化引擎实例"""
    if db_path is None:
        from hermes_constants import get_hermes_home
        hermes_home = get_hermes_home()
        db_path = str(Path(hermes_home) / "evo_master.db")
    
    cache = KnowledgeCache(db_path)
    return EvoMaster(cache, lambda_weight)


if __name__ == "__main__":
    # 测试
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    print("=" * 70)
    print("EvoMaster 测试")
    print("=" * 70)
    print()
    
    # 创建进化引擎
    evo = get_evo_master(db_path, lambda_weight=0.3)
    print(f"✅ 初始化 EvoMaster (λ=0.3)")
    print()
    
    # 记录轨迹
    print("📝 记录执行轨迹...")
    trajectories = [
        ("web_search python", [{"action": "search", "query": "python"}], True, 0.9),
        ("web_search rust", [{"action": "search", "query": "rust"}], True, 0.85),
        ("file_read test.py", [{"action": "read", "path": "test.py"}], True, 0.8),
        ("web_search failed", [{"action": "search", "query": "xxx"}], False, 0.0),
    ]
    
    for task, actions, success, reward in trajectories:
        traj = evo.record_trajectory(task, actions, success, reward)
        print(f"  • {task}: 价值={traj.total_value:.2f} (R={reward:.2f}, K={traj.knowledge_value:.2f})")
    print()
    
    # 策略进化
    print("🔄 策略进化...")
    new_strategy = evo.evolve_strategy()
    print(f"  新策略: v{new_strategy.version}, 性能={new_strategy.performance:.2f}")
    print()
    
    # 推荐动作
    print("💡 推荐动作...")
    task = "web_search python tutorial"
    recommended = evo.get_recommended_actions(task)
    print(f"  任务: {task}")
    print(f"  推荐: {recommended}")
    print()
    
    # 统计
    top_trajs = evo.knowledge_cache.get_top_trajectories(limit=3)
    print(f"📊 最优轨迹 (Top 3):")
    for i, t in enumerate(top_trajs, 1):
        print(f"  {i}. {t.task}: 价值={t.total_value:.2f}")
    print()
    
    print("=" * 70)
    print("✅ EvoMaster 测试完成")
    print("=" * 70)
