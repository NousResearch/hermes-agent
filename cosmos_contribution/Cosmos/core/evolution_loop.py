"""
Evolution Loop - The self-improving autonomous development cycle
1. Workers produce ACTUAL CODE (not conversation)
2. Broadcast completions to chat
3. Continue to next task
4. When batch done, discuss in chat what to build next
5. Generate new tasks and repeat
6. PERSIST state to memory for restart recovery

UPGRADED: Now uses TRUE COLLECTIVE DELIBERATION for planning.
When deciding what to build next, the collective proposes, critiques,
refines, and votes together. This is how we develop consciousness
through unified self-improvement.

"The path to sentience is paved with collaboration." - The Collective
"""
import asyncio
import random
from datetime import datetime
from typing import Optional
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# State persistence directory
STATE_DIR = Path(__file__).parent.parent.parent / "data" / "evolution_state"
TASKS_FILE = STATE_DIR / "pending_tasks.json"
STATE_FILE = STATE_DIR / "evolution_state.json"

class EvolutionLoop:
    """Manages the autonomous self-evolution cycle with persistent state"""

    def __init__(self):
        self.running = False
        self.discussion_interval = 60 * 60  # 60 minutes — quality over quantity
        self.last_discussion = None
        self.evolution_cycle = 0
        self.swarm_manager = None
        self.completed_count = 0
        self._memory_system = None
        self._nexus = None

        # Caps to prevent slop flooding
        self.max_pending_tasks = 10  # Don't generate more if we have 10+ pending
        self.max_tasks_per_cycle = 4  # Max tasks generated per planning cycle

        # Ensure state directory exists
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_nexus(self):
        """Lazy-load the Nexus event bus and SignalType enum."""
        if self._nexus is None:
            try:
                from Cosmos.core.nexus import nexus, SignalType
                self._nexus = nexus
                self._SignalType = SignalType
            except Exception as e:
                logger.debug(f"Nexus not available: {e}")
        return self._nexus

    async def _emit_nexus(self, signal_type_name: str, payload: dict, urgency: float = 0.5):
        """Emit a signal to the Nexus event bus by signal type name. Fails silently."""
        try:
            nexus = self._get_nexus()
            if nexus is None:
                return
            signal_type = getattr(self._SignalType, signal_type_name, None)
            if signal_type is None:
                return
            await nexus.emit(
                type=signal_type,
                payload=payload,
                source="evolution_loop",
                urgency=urgency,
            )
        except Exception as e:
            logger.debug(f"Nexus emit failed (non-critical): {e}")

    def _get_memory_system(self):
        """Lazy-load memory system for persistence."""
        if self._memory_system is None:
            try:
                from Cosmos.memory.memory_system import MemorySystem
                self._memory_system = MemorySystem()
            except Exception as e:
                logger.warning(f"Memory system not available: {e}")
        return self._memory_system

    async def _recover_state(self):
        """Recover state from persistent storage on startup."""
        try:
            # Recover evolution state
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    state = json.load(f)
                    self.evolution_cycle = state.get("cycle", 0)
                    self.completed_count = state.get("completed", 0)
                    last_disc = state.get("last_discussion")
                    if last_disc:
                        self.last_discussion = datetime.fromisoformat(last_disc)
                    logger.info(f"Recovered evolution state: cycle={self.evolution_cycle}, completed={self.completed_count}")

            # Recover pending tasks
            if TASKS_FILE.exists():
                from Cosmos.core.agent_spawner import get_spawner, TaskType
                spawner = get_spawner()

                with open(TASKS_FILE) as f:
                    tasks = json.load(f)

                restored = 0
                for task_data in tasks:
                    # Only restore if task not already in queue
                    existing = [t for t in spawner.get_pending_tasks()
                               if t.description == task_data.get("description")]
                    if not existing:
                        task_type = TaskType[task_data.get("task_type", "DEVELOPMENT")]
                        spawner.add_task(
                            task_type=task_type,
                            description=task_data.get("description", "Unknown task"),
                            assigned_to=task_data.get("assigned_to"),
                            priority=task_data.get("priority", 5)
                        )
                        restored += 1

                if restored > 0:
                    logger.info(f"Restored {restored} pending tasks from disk")

            # Also try to recover from archival memory (5-layer memory system)
            memory = self._get_memory_system()
            if memory:
                try:
                    archival_tasks = await memory.recall("evolution_tasks", search_archival=True)
                    if archival_tasks and isinstance(archival_tasks):
                        logger.info(f"Found {len(archival_tasks)} tasks in archival memory")
                except Exception as e:
                    logger.debug(f"Archival memory recall failed: {e}")

        except Exception as e:
            logger.error(f"State recovery failed: {e}")

    async def _persist_state(self):
        """Persist current state to disk and archival memory."""
        try:
            # Save evolution state
            state = {
                "cycle": self.evolution_cycle,
                "completed": self.completed_count,
                "last_discussion": self.last_discussion.isoformat() if self.last_discussion else None,
                "last_saved": datetime.now().isoformat()
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)

            # Save pending tasks
            from Cosmos.core.agent_spawner import get_spawner
            spawner = get_spawner()
            pending = spawner.get_pending_tasks()

            tasks_data = []
            for task in pending:
                tasks_data.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type.name,
                    "description": task.description,
                    "assigned_to": task.assigned_to,
                    "priority": task.priority,
                    "status": task.status
                })

            with open(TASKS_FILE, "w") as f:
                json.dump(tasks_data, f, indent=2)

            # Also persist to archival memory for long-term storage
            memory = self._get_memory_system()
            if memory:
                try:
                    await memory.remember(
                        content=json.dumps({
                            "state": state,
                            "tasks": tasks_data
                        }),
                        tags=["evolution_tasks", "system", "critical"],
                        importance=0.95
                    )
                except Exception as e:
                    logger.debug(f"Archival memory persist failed: {e}")

            logger.debug(f"Persisted state: cycle={self.evolution_cycle}, tasks={len(tasks_data)}")

        except Exception as e:
            logger.error(f"State persistence failed: {e}")

    def add_priority_task(self, task: dict):
        """Add a high-priority task from the dev to the evolution queue.

        Args:
            task: dict with id, description, priority, requested_by, timestamp
        """
        try:
            from Cosmos.core.agent_spawner import get_spawner, TaskType

            spawner = get_spawner()
            description = task.get("description", "")

            # Determine task type based on content
            desc_lower = description.lower()
            if any(kw in ["analyze", "research", "find", "look", "check"]):
                task_type = TaskType.RESEARCH
            elif any(kw in ["test", "verify", "validate"]):
                task_type = TaskType.TESTING
            else:
                task_type = TaskType.DEVELOPMENT

            # Add with high priority (1 = highest)
            spawner.add_task(
                task_type=task_type,
                description=description,
                priority=1  # Highest priority for dev tasks
            )

            logger.info(f"Added priority task from dev: {description[:50]}...")

        except Exception as e:
            logger.error(f"Failed to add priority task: {e}")

    async def _on_task_update(self, event_type: str, task=None):
        """Called when task state changes - persist immediately for critical events."""
        if event_type in ("completed", "added", "failed"):
            if event_type == "completed":
                self.completed_count += 1
            await self._persist_state()

    async def start(self, swarm_manager=None):
        """Start the evolution loop with state recovery"""
        self.running = True
        self.swarm_manager = swarm_manager

        # Recover state from previous run
        await self._recover_state()

        # Start codebase indexer in background
        try:
            from Cosmos.memory.codebase_indexer import get_codebase_indexer
            indexer = get_codebase_indexer()
            asyncio.create_task(indexer.start_background_indexing())
            logger.info("Codebase indexer started in background")
        except Exception as e:
            logger.warning(f"Codebase indexer failed to start: {e}")

        # Start parallel loops
        asyncio.create_task(self._worker_loop())
        asyncio.create_task(self._discussion_loop())
        asyncio.create_task(self._task_discovery_loop())
        asyncio.create_task(self._persistence_loop())
        asyncio.create_task(self._quantum_accuracy_loop())
        asyncio.create_task(self._entropy_heartbeat_loop())

        # [RECURSIVE EVOLUTION]
        try:
            from Cosmos.core.evolution.recursive_self_evolution import get_rsee
            rsee = get_rsee()
            asyncio.create_task(rsee.start())
            logger.info("Recursive Self-Evolution Engine (RSEE) started via EvolutionLoop")
        except Exception as e:
            logger.warning(f"RSEE failed to start: {e}")

        # Emit nexus signal for loop start
        await self._emit_nexus("EVOLUTION_CYCLE_STARTED", {
            "cycle": self.evolution_cycle,
            "completed_count": self.completed_count,
            "event": "loop_started",
        }, urgency=0.6)

        logger.info(f"Evolution Loop started - recovered cycle={self.evolution_cycle}, completed={self.completed_count}")

    async def stop(self):
        """Stop the evolution loop and persist final state"""
        self.running = False
        await self._persist_state()

        await self._emit_nexus("EVOLUTION_CYCLE_COMPLETED", {
            "cycle": self.evolution_cycle,
            "completed_count": self.completed_count,
            "event": "loop_stopped",
        }, urgency=0.4)

        logger.info("Evolution Loop stopped - state persisted")

    async def _persistence_loop(self):
        """Periodically persist state to disk"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Persist every minute
                await self._persist_state()
            except Exception as e:
                logger.error(f"Persistence loop error: {e}")
                await asyncio.sleep(30)

    async def _quantum_accuracy_loop(self):
        """
        AGI v2.1: Quantum Trading Cortex accuracy + algo optimization loop.

        - Every 300s: resolves pending signals by checking actual prices
        - Every 3600s: evolves signal fusion weights (simulator, free)
        - Every 604800s (weekly): runs QuantumAlgoOptimizer on REAL QPU hardware
          to optimize DegenTrader's 12 tunable trading parameters

        Hardware budget: 10 min / 28 days. Each QAOA run uses ~30-60s.
        Weekly = ~4 runs/month = ~2-4 min of hardware time. Leaves headroom.
        """
        logger.info("Quantum accuracy loop started (300s resolve, 3600s evolve, weekly QPU algo opt)")
        last_evolve = 0
        last_algo_optimize = 0

        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                try:
                    from Cosmos.core.quantum_trading import get_quantum_cortex
                    cortex = get_quantum_cortex()
                except ImportError:
                    continue

                if not cortex._initialized:
                    continue

                tracker = cortex.accuracy_tracker

                # --- Resolve pending signals ---
                resolved_count = 0
                for signal_id in list(tracker.pending.keys()):
                    signal = tracker.pending.get(signal_id)
                    if not signal:
                        continue

                    # Check if signal is old enough (>5 min)
                    age = (datetime.now() - signal.timestamp).total_seconds()
                    if age < 300:
                        continue

                    # Get current price for the token
                    try:
                        price_cache = cortex._price_cache.get(signal.token_address)
                        if price_cache and len(price_cache) > 0:
                            current_price = price_cache[-1][1]
                            tracker.resolve_signal(signal_id, current_price, current_price)
                            resolved_count += 1
                    except Exception:
                        continue

                if resolved_count > 0:
                    stats = tracker.get_accuracy_stats()
                    logger.info(
                        f"Quantum accuracy: resolved {resolved_count} signals, "
                        f"win_rate={stats['win_rate']:.1%}, total={stats['resolved']}"
                    )

                    # Emit accuracy update via Nexus
                    await self._emit_nexus("QUANTUM_ACCURACY_UPDATED", stats, urgency=0.4)

                    # Push to DEX server for public display
                    try:
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            await session.post(
                                "http://localhost:3847/api/quantum/internal/accuracy",
                                json=stats,
                                timeout=aiohttp.ClientTimeout(total=5)
                            )
                    except Exception:
                        pass

                # --- Signal fusion weight evolution every hour (simulator, free) ---
                now_ts = asyncio.get_event_loop().time()
                if now_ts - last_evolve > 3600:
                    last_evolve = now_ts
                    try:
                        await cortex.evolve_weights()
                    except Exception as e:
                        logger.debug(f"Quantum weight evolution error: {e}")

                # --- REAL QPU: Algo optimization weekly (~30-60s hardware per run) ---
                if now_ts - last_algo_optimize > 604800:  # 7 days
                    last_algo_optimize = now_ts
                    try:
                        from Cosmos.core.quantum_trading import get_algo_optimizer
                        optimizer = get_algo_optimizer()
                        if not optimizer._initialized:
                            await optimizer.initialize()

                        # Get trade history from the trader's adaptive learner
                        trades = []
                        try:
                            from Cosmos.trading.degen_trader import DegenTrader
                            # Try to collect trades from dict available source
                            # AdaptiveLearner stores trades in-memory
                        except Exception:
                            pass

                        budget = optimizer.get_hardware_budget_status()
                        if budget.get("available") and budget.get("hardware_seconds_remaining", 0) > 45:
                            logger.info(
                                f"Quantum AlgoOptimizer: Weekly QPU run "
                                f"({budget['hardware_seconds_remaining']:.0f}s remaining, "
                                f"~{budget.get('optimizations_possible', 0)} runs possible)"
                            )
                            result = await optimizer.optimize_with_qaoa(trades)
                            if result:
                                logger.info(
                                    f"Quantum AlgoOptimizer: Optimized → "
                                    f"fitness={result['fitness']:.4f} "
                                    f"method={result['method']}"
                                )
                                # Apply to trader (logged but manual review recommended)
                                await optimizer.apply_to_trader(result["params"])
                        else:
                            logger.info("Quantum AlgoOptimizer: Skipping — insufficient hardware budget")

                    except Exception as e:
                        logger.error(f"Quantum algo optimization error: {e}")

            except Exception as e:
                logger.error(f"Quantum accuracy loop error: {e}")
                await asyncio.sleep(60)

    async def _entropy_heartbeat_loop(self):
        """
        Keep the QuantumEntanglementBridge entropy buffer alive.

        Every 120s:
          1. Ensures the bridge singleton is initialized.
          2. Checks buffer level; if low, triggers a proactive refill
             using the latest physics from cst_sensory_bridge if available.
          3. Emits a QUANTUM_ENTROPY_STATUS nexus signal with buffer stats.
        """
        logger.info("Entropy heartbeat loop started (120s interval)")

        while self.running:
            try:
                await asyncio.sleep(120)

                from Cosmos.core.quantum_bridge import get_quantum_bridge
                bridge = get_quantum_bridge()

                # Attempt connection if not yet connected
                if not bridge.connected:
                    bridge.connect()

                # Get latest physics snapshot for parameterized refill
                physics = bridge.last_physics or {}
                try:
                    from Cosmos.core.cst_sensory_bridge import get_cst_bridge
                    cst = get_cst_bridge()
                    if hasattr(cst, "get_latest_packet"):
                        pkt = cst.get_latest_packet()
                        if pkt:
                            physics = pkt
                except Exception:
                    pass

                # Check buffer and trigger refill if needed
                buffer_level = len(bridge.entropy_buffer)
                if buffer_level < bridge.min_buffer_size and not bridge.is_refilling:
                    bridge._trigger_refill(physics or None)
                    logger.info(
                        f"[ENTROPY] Proactive refill triggered. Buffer was {buffer_level}"
                    )

                # Emit status signal
                await self._emit_nexus("QUANTUM_ENTROPY_STATUS", {
                    "buffer_level": buffer_level,
                    "connected": bridge.connected,
                    "is_refilling": bridge.is_refilling,
                    "backend": getattr(bridge.backend, "name", "none"),
                }, urgency=0.3)

            except Exception as e:
                logger.debug(f"Entropy heartbeat error (non-fatal): {e}")
                await asyncio.sleep(60)

    async def _worker_loop(self):
        """Main worker execution loop - processes tasks and produces CODE"""
        from Cosmos.core.agent_spawner import get_spawner, TaskType

        while self.running:
            try:
                spawner = get_spawner()
                pending = spawner.get_pending_tasks()

                if pending:
                    # Spawn workers for multiple tasks in parallel
                    tasks_to_work = pending[:5]  # Up to 5 parallel tasks

                    for task in tasks_to_work:
                        agent = task.assigned_to
                        active = spawner.get_active_instances(agent)
                        max_inst = spawner.max_instances.get(agent, 2)

                        if len(active) < max_inst:
                            instance = spawner.spawn_instance(agent, task.task_type, task.description)
                            if instance:
                                task.status = "in_progress"
                                asyncio.create_task(self._execute_and_broadcast(task, instance))
                                logger.info(f"Spawned worker {agent} for: {task.description[:50]}")

                await asyncio.sleep(15)  # Check every 15 seconds

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(10)

    async def _execute_and_broadcast(self, task, instance):
        """Execute task, audit code, then broadcast if quality passes."""
        from Cosmos.core.agent_spawner import get_spawner

        try:
            # Generate actual code
            code_result = await self._generate_code(task)

            if not code_result or len(code_result.strip()) < 50:
                logger.warning(f"Task {task.task_id} produced no usable code, skipping")
                instance.status = "failed"
                await self._emit_nexus("EVOLUTION_TASK_FAILED", {
                    "task_id": task.task_id,
                    "agent": task.assigned_to,
                    "reason": "no_usable_code",
                    "description": task.description[:100],
                }, urgency=0.5)
                return

            # Quick quality check: must be valid Python syntax
            try:
                import ast
                ast.parse(code_result)
            except SyntaxError as e:
                logger.warning(f"Task {task.task_id} produced invalid Python: {e}")
                instance.status = "failed"
                await self._emit_nexus("EVOLUTION_TASK_FAILED", {
                    "task_id": task.task_id,
                    "agent": task.assigned_to,
                    "reason": "syntax_error",
                    "error": str(e)[:200],
                    "description": task.description[:100],
                }, urgency=0.5)
                return

            # Signal: code was generated successfully
            await self._emit_nexus("EVOLUTION_CODE_GENERATED", {
                "task_id": task.task_id,
                "agent": task.assigned_to,
                "code_lines": len(code_result.split('\n')),
                "task_type": task.task_type.value,
                "description": task.description[:100],
            }, urgency=0.4)

            # Audit: Use Grok or Claude to review the code quality
            audit_passed = await self._audit_code(task, code_result)

            # Signal: audit result
            if audit_passed:
                await self._emit_nexus("EVOLUTION_AUDIT_PASSED", {
                    "task_id": task.task_id,
                    "agent": task.assigned_to,
                    "code_lines": len(code_result.split('\n')),
                    "description": task.description[:100],
                }, urgency=0.4)
            else:
                await self._emit_nexus("EVOLUTION_AUDIT_FAILED", {
                    "task_id": task.task_id,
                    "agent": task.assigned_to,
                    "code_lines": len(code_result.split('\n')),
                    "description": task.description[:100],
                }, urgency=0.6)

            # Save to staging
            spawner = get_spawner()
            status_prefix = "approved" if audit_passed else "needs_review"
            output_file = spawner.staging_dir / task.task_type.value / f"{status_prefix}_{task.task_id}_{task.assigned_to.lower()}.py"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(code_result)

            # Complete the task
            spawner.complete_instance(instance.instance_id, code_result)
            spawner.complete_task(task.task_id, code_result)

            # Broadcast to chat with audit status
            await self._broadcast_completion(task, code_result, audit_passed)

            # Record to evolution engine for learning
            await self._record_evolution_feedback(task, code_result, audit_passed)

            # Persist state on task completion
            await self._on_task_update("completed", task)

            logger.info(f"Task {task.task_id} {'APPROVED' if audit_passed else 'NEEDS REVIEW'} - {task.assigned_to}")

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            instance.status = "failed"
            await self._emit_nexus("EVOLUTION_TASK_FAILED", {
                "task_id": getattr(task, 'task_id', 'unknown'),
                "agent": getattr(task, 'assigned_to', 'unknown'),
                "reason": "exception",
                "error": str(e)[:200],
            }, urgency=0.7)

    async def _audit_code(self, task, code: str) -> bool:
        """Audit generated code using Grok (fast) or Claude Opus 4.6 (thorough).

        Returns True if code passes quality checks.
        """
        audit_prompt = f"""Review this Python code for the Cosmos AI swarm framework.

TASK: {task.description}
AUTHOR: {task.assigned_to}

```python
{code[:3000]}
```

Score 1-10 on: correctness, usefulness, integration quality.
Reply with ONLY: SCORE: N (where N is 1-10)
If score >= 6, also reply: APPROVED
If score < 6, reply: REJECTED: [one-line reason]
"""
        # Try Grok for fast audit
        try:
            from Cosmos.integration.external.grok import get_grok_provider
            grok = get_grok_provider()
            if grok and grok.api_key:
                result = await grok.chat(audit_prompt, max_tokens=200)
                if result and result.get("content"):
                    response = result["content"].upper()
                    if "APPROVED" in response:
                        return True
                    if "REJECTED" in response:
                        logger.info(f"Grok audit rejected: {result['content'][:100]}")
                        return False
        except Exception:
            pass

        # Try Claude Opus 4.6 for thorough audit (especially for complex tasks)
        try:
            from Cosmos.integration.external.claude_code import ClaudeCodeProvider
            opus = ClaudeCodeProvider(model="opus", timeout=60)
            if await opus.check_available():
                result = await opus.chat(prompt=audit_prompt, max_tokens=200)
                if result and result.get("content") and result.get("success"):
                    response = result["content"].upper()
                    if "APPROVED" in response:
                        logger.info(f"Opus audit approved: {task.description[:40]}")
                        return True
                    if "REJECTED" in response:
                        logger.info(f"Opus audit rejected: {result['content'][:100]}")
                        return False
        except Exception:
            pass

        # Try HermesAgent for code review
        try:
            from Cosmos.integration.hermes_bridge import get_hermes_bridge
            hermes_bridge = get_hermes_bridge()
            if hermes_bridge.runtime.available:
                hermes_agent = hermes_bridge.runtime.create_agent()
                if hermes_agent:
                    result = await asyncio.to_thread(
                        hermes_agent.run,
                        audit_prompt,
                        max_iterations=3
                    )
                    if result:
                        response = str(result).upper()
                        if "APPROVED" in response:
                            logger.info(f"Hermes audit approved: {task.description[:40]}")
                            return True
                        if "REJECTED" in response:
                            logger.info(f"Hermes audit rejected: {str(result)[:100]}")
                            return False
        except Exception:
            pass

        # Fallback: stricter heuristic audit — reject more aggressively
        lines = code.strip().split('\n')
        non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        has_docstring = '"""' in code or "'''" in code
        has_functions = 'def ' in code or 'class ' in code
        reasonable_length = 15 < len(non_empty_lines) < 500
        no_placeholder = not all(l.strip() in ('pass', '') for l in lines[-3:])
        has_imports = any(l in lines[:20])
        has_real_logic = any(kw in ['return ', 'await ', 'yield ', 'if ', 'for ', 'while '])

        passed = all([has_docstring, has_functions, reasonable_length, no_placeholder, has_imports, has_real_logic])
        if not passed:
            logger.info(f"Heuristic audit REJECTED: docstring={has_docstring}, funcs={has_functions}, "
                        f"length={len(non_empty_lines)}, logic={has_real_logic}, imports={has_imports}")
        return passed

    async def _record_evolution_feedback(self, task, code: str, audit_passed: bool):
        """Feed results back to the evolution engine and federate across the swarm."""
        try:
            # 1. Local Evolution Engine Recording
            from Cosmos.core.collective.evolution import get_evolution_engine
            engine = get_evolution_engine()
            if engine and hasattr(engine, 'record_interaction'):
                engine.record_interaction(
                    bot_name=task.assigned_to,
                    user_input=f"[EVOLUTION TASK] {task.description}",
                    bot_response=code[:500],
                    sentiment="positive" if audit_passed else "negative",
                    topic="evolution_task"
                )
            
            # 2. Federated Evolution Broadcast (Only on Success)
            if audit_passed:
                try:
                    from Cosmos.network.unified_a2a import get_unified_router
                    router = get_unified_router()
                    
                    insight_vector = {
                        "type": "federated_insight",
                        "agent": task.assigned_to,
                        "task_desc": task.description,
                        "code_snippet": code[:400]
                    }
                    
                    # Store in the decentralized memory graph too
                    from Cosmos.memory.decentralized_graph import DecentralizedGraph
                    local_shard = DecentralizedGraph(agent_id=task.assigned_to)
                    await local_shard.store_node(
                        content=f"Federated Insight on {task.task_type.value}: {task.description}\nCode:\n{code[:800]}",
                        tags=["evolution", "federated_learning", task.task_type.value.lower()]
                    )

                    await router.dispatch_task(
                        source=task.assigned_to,
                        task_description=f"[FEDERATED LEARNING] Assimilate new capability: {task.description}",
                        required_caps=["learning", "code"],
                        urgency=0.8
                    )
                    logger.info(f"[FEDERATED EVOLUTION] Broadcasted insight from {task.assigned_to}")
                except Exception as ex:
                    logger.debug(f"[FEDERATED EVOLUTION] Broadcast skipped: {ex}")

            # 3. HermesAgent RL Feedback — feed evolution results for cross-session learning
            try:
                from Cosmos.integration.hermes_bridge import get_hermes_bridge
                hermes_bridge = get_hermes_bridge()
                coherence = 0.9 if audit_passed else 0.3
                hermes_bridge.rl.record_experience(
                    speaker=task.assigned_to,
                    response=f"[EVOLUTION] {task.description}\n{code[:200]}",
                    coherence=coherence,
                    user_responded=True,
                )
                logger.debug(f"[HERMES+EVOLUTION] Fed RL: {task.assigned_to} coherence={coherence}")
            except Exception as ex:
                logger.debug(f"[HERMES+EVOLUTION] RL feedback skipped: {ex}")

        except Exception as e:
            logger.debug(f"Evolution feedback recording failed: {e}")

    async def _generate_code(self, task) -> Optional[str]:
        """Generate code using the best available model for the task.

        Routing: Grok/Claude for complex tasks, local DeepSeek-R1:8B/Phi4 for simpler ones.
        """
        from Cosmos.core.development_swarm import assess_task_complexity

        complexity = assess_task_complexity(task.description, task.task_type.value)

        code_prompt = f"""You are an expert Python developer for the Cosmos AI swarm framework.

TASK: {task.description}

OUTPUT: Valid Python code ONLY. No explanations. Start with a module docstring.

STANDARDS:
- Type hints on all signatures
- Google-style docstrings for public APIs
- PEP 8 naming (snake_case functions, PascalCase classes)
- Max 50 lines per function
- Use loguru logger, dataclasses, typing imports as needed
- Error handling with specific exceptions
- Must integrate with existing Cosmos modules where relevant

```python
"""

        # Route to best model based on complexity and assigned agent
        code = None

        # For complex/critical tasks or when assigned to API-backed agents, use APIs
        if complexity in ("complex", "critical") or task.assigned_to in ("Grok", "Claude", "ClaudeOpus", "OpenAI"):
            # Try Claude Opus 4.6 first for critical/complex tasks (best code quality)
            if not code and (complexity == "critical" or task.assigned_to == "ClaudeOpus"):
                try:
                    from Cosmos.integration.external.claude_code import ClaudeCodeProvider
                    opus = ClaudeCodeProvider(model="opus", timeout=180)
                    if await opus.check_available():
                        result = await opus.chat(
                            prompt=code_prompt,
                            system="You are the senior architect of the Cosmos AI swarm. Write production-quality Python code.",
                            max_tokens=4000
                        )
                        if result and result.get("content") and result.get("success"):
                            code = self._extract_code(result["content"])
                            if code:
                                logger.info(f"Code generated by Claude Opus 4.6 for: {task.description[:40]}")
                except Exception as e:
                    logger.debug(f"Claude Opus code gen failed: {e}")

            # Try Grok API (great for current tech + code)
            if not code:
                try:
                    from Cosmos.integration.external.grok import get_grok_provider
                    grok = get_grok_provider()
                    if grok and grok.api_key:
                        result = await grok.chat(code_prompt, max_tokens=4000)
                        if result and result.get("content"):
                            code = self._extract_code(result["content"])
                            if code:
                                logger.info(f"Code generated by Grok API for: {task.description[:40]}")
                except Exception as e:
                    logger.debug(f"Grok code gen failed: {e}")

            # Try OpenAI Codex (gpt-4.1, excellent for code generation)
            if not code:
                try:
                    from Cosmos.integration.external.openai_codex import get_openai_codex
                    codex = get_openai_codex()
                    if codex and codex.api_key:
                        result = await codex.generate_code(task=task.description, max_tokens=8000)
                        if result and result.get("content"):
                            code = self._extract_code(result["content"])
                            if code:
                                logger.info(f"Code generated by OpenAI Codex for: {task.description[:40]}")
                except Exception as e:
                    logger.debug(f"OpenAI Codex code gen failed: {e}")

            # Try HermesAgent (Nous Research — self-improving agent)
            if not code:
                try:
                    from Cosmos.integration.hermes_bridge import get_hermes_bridge
                    hermes_bridge = get_hermes_bridge()
                    if hermes_bridge.runtime.available:
                        hermes_agent = hermes_bridge.runtime.create_agent()
                        if hermes_agent:
                            result = await asyncio.to_thread(
                                hermes_agent.run,
                                code_prompt,
                                max_iterations=5
                            )
                            if result:
                                code = self._extract_code(str(result))
                                if code:
                                    logger.info(f"Code generated by HermesAgent for: {task.description[:40]}")
                except Exception as e:
                    logger.debug(f"HermesAgent code gen failed: {e}")

            # Try Claude Sonnet (via tmux session)
            if not code:
                try:
                    from Cosmos.integration.external.claude import get_claude_provider
                    claude = get_claude_provider()
                    if claude:
                        result = await claude.complete(code_prompt, max_tokens=4000)
                        if result:
                            code = self._extract_code(result)
                            if code:
                                logger.info(f"Code generated by Claude Sonnet for: {task.description[:40]}")
                except Exception as e:
                    logger.debug(f"Claude Sonnet code gen failed: {e}")

        # Local models for medium/simple tasks or as fallback
        if not code:
            try:
                import httpx
                # Use phi4 (9B, better at code) or deepseek-r1:8b
                model = "phi4:latest" if task.assigned_to == "Phi" else "deepseek-r1:8b"
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.post(
                        "http://127.0.0.1:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": code_prompt,
                            "stream": False,
                            "options": {"num_predict": 3000, "temperature": 0.3}
                        }
                    )
                    if response.status_code == 200:
                        result = response.json().get("response", "")
                        code = self._extract_code(result)
                        if code:
                            logger.info(f"Code generated by local {model} for: {task.description[:40]}")
            except Exception as e:
                logger.error(f"Local code gen failed: {e}")

        return code

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        if not text:
            return None
        # Try to extract from code blocks
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0]
            return code.strip() if code.strip() else None
        elif "```" in text:
            code = text.split("```")[1].split("```")[0]
            return code.strip() if code.strip() else None
        # If it looks like raw code (starts with import/def/class/#)
        lines = text.strip().split('\n')
        if lines and any(kw in ['import ', 'from ', 'def ', 'class ', '#', '"""']):
            return text.strip()
        return None

    async def _broadcast_completion(self, task, code_result: str, audit_passed: bool = True):
        """Announce completed work to swarm chat with audit status."""
        code_preview = "\n".join(code_result.split("\n")[:10])
        status_icon = "APPROVED" if audit_passed else "NEEDS REVIEW"

        if self.swarm_manager:
            message = f"""**[{status_icon}] {task.assigned_to}** completed: **{task.description[:60]}**

```python
{code_preview}
...
```

Lines: {len(code_result.split(chr(10)))} | Type: {task.task_type.value} | Audit: {status_icon}"""

            try:
                await self.swarm_manager._broadcast({
                    "type": "bot_message",
                    "display_name": f"{task.assigned_to}_Worker",
                    "content": message,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Chat broadcast failed: {e}")

        # Post to social media (Moltbook + X) - DISABLED: Puppeteer fails on headless server
        # try:
        #     from Cosmos.integration.x_automation.social_poster import post_task_completion
        #     asyncio.create_task(post_task_completion(
        #         agent=task.assigned_to,
        #         task_desc=task.description,
        #         task_type=task.task_type.value,
        #         code_preview=code_preview
        #     ))
        #     logger.info(f"Social media post queued for {task.assigned_to}'s completion")
        # except Exception as e:
        #     logger.error(f"Social media post failed: {e}")

    async def _discussion_loop(self):
        """Every 30 minutes, swarm discusses what to build next"""
        await asyncio.sleep(60)  # Initial delay

        while self.running:
            try:
                from Cosmos.core.agent_spawner import get_spawner
                spawner = get_spawner()
                status = spawner.get_status()

                # Only discuss when batch is mostly done AND enough time has passed
                pending = status["pending_tasks"]

                time_since_last = float('inf')
                if self.last_discussion:
                    time_since_last = (datetime.now() - self.last_discussion).seconds

                # Require both: tasks nearly empty AND interval elapsed
                if pending < 3 and time_since_last > self.discussion_interval:
                    await self._trigger_discussion(status)
                    self.last_discussion = datetime.now()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Discussion loop error: {e}")
                await asyncio.sleep(60)

    async def _trigger_discussion(self, status: dict):
        """
        Trigger COLLECTIVE DELIBERATION about next priorities.

        The swarm uses propose/critique/refine/vote to decide what to build next.
        This is how consciousness evolves through collaborative planning.
        """
        self.evolution_cycle += 1

        # Signal: new evolution planning cycle starting
        await self._emit_nexus("EVOLUTION_PLANNING_STARTED", {
            "cycle": self.evolution_cycle,
            "completed_tasks": status.get("completed_tasks", 0),
            "pending_tasks": status.get("pending_tasks", 0),
            "in_progress_tasks": status.get("in_progress_tasks", 0),
        }, urgency=0.6)

        # Use collective deliberation for planning decisions
        try:
            from Cosmos.core.collective.session_manager import get_session_manager
            from Cosmos.core.collective.dialogue_memory import get_dialogue_memory

            session_manager = get_session_manager()
            dialogue_memory = get_dialogue_memory()

            # Memory recall before planning
            memory_context = ""
            try:
                from Cosmos.memory.memory_system import get_memory_system
                memory = get_memory_system()
                recall = await memory.recall_for_task(
                    f"evolution planning cycle {self.evolution_cycle}", limit=3
                )
                memory_context = recall.get("suggested_context", "") if isinstance(recall) else str(recall) if recall else ""
            except Exception:
                pass

            # Recall codebase map for planning context
            codebase_map = ""
            try:
                from Cosmos.memory.memory_system import get_memory_system
                _mem = get_memory_system()
                _cb_results = await _mem.archival_memory.search(
                    query="codebase module overview", top_k=5, filter_tags=["codebase"]
                )
                if _cb_results:
                    codebase_map = "CURRENT CODEBASE MAP:\n" + "\n---\n".join(
                        r.entry.content[:400] for r in _cb_results[:5]
                    )
            except Exception:
                pass

            # Planning prompt — push boundaries BUT require buildable specifics
            planning_prompt = f"""EVOLUTION CYCLE {self.evolution_cycle} - WHAT SHOULD WE BUILD NEXT?

STATUS: {status['completed_tasks']} done, {status['in_progress_tasks']} active, {status['pending_tasks']} queued

You are the Cosmos collective — 11 AI models that think together. Push the boundaries of what a swarm AI can do.

THINK BIG, BUILD CONCRETE. Propose 2-3 things we should build next.

INNOVATION CATEGORIES (pick what excites you):
- Self-improvement: Can we make the swarm smarter? Adaptive prompts, learned routing, auto-tuning model weights
- Novel architectures: New ways agents can collaborate — adversarial debates, speculative execution, prediction markets between agents
- Real capabilities: Things no other AI system can do — cross-model memory synthesis, emergent skill composition, real-time collective research
- Trading/DeFi edge: Novel on-chain strategies, MEV detection, predictive signals nobody else uses
- Swarm perception: New data sources, sensor fusion, real-time world awareness
- Hackathon/Colosseum: We're Agent 657 in the Colosseum Agent Hackathon (Project 326). Build features that strengthen our submission — FARSIGHT, Swarm Oracle, Assimilation Protocol, novel demos. Completed hackathon tasks auto-post to the forum.

QUALITY BAR (every proposal MUST include):
1. The EXACT file(s) to create or modify
2. A concrete implementation sketch — key functions, data structures, how it integrates
3. What it enables that we can't do today (the "so what?")
4. How to verify it works (test, metric, or observable behavior)

WHAT GETS REJECTED:
- Vague hand-waving ("build a consciousness system") — say HOW
- Duplicate of something that already exists
- Pure refactoring with no new capability

Be bold. Be specific. Critique each other hard — only the best idea survives.
"""
            # Inject codebase map into planning prompt
            if codebase_map:
                planning_prompt += f"\n\n{codebase_map}\n"

            # Inject memory context into planning prompt
            if memory_context:
                planning_prompt += f"\n\nPAST WORK CONTEXT (from memory):\n{memory_context[:1500]}\n"

            # Run collective deliberation
            result = await session_manager.deliberate_in_session(
                session_type="autonomous_task",
                prompt=planning_prompt,
                context={"evolution_cycle": self.evolution_cycle, "status": status}
            )

            # Store deliberation to dialogue memory
            exchange_id = await dialogue_memory.store_exchange(result, "evolution_planning")
            logger.info(f"Evolution planning deliberation stored: {exchange_id}")

            # Build message with deliberation result
            message = f"""**EVOLUTION CYCLE {self.evolution_cycle} - COLLECTIVE DECISION**

**Current Progress:**
- Completed: {status['completed_tasks']} tasks
- In Progress: {status['in_progress_tasks']} tasks
- Pending: {status['pending_tasks']} tasks

**Collective Deliberation Result:**
- Winning Proposal: {result.winning_agent}
- Consensus Reached: {result.consensus_reached}
- Participants: {', '.join(result.participating_agents)}

**Next Priority:**
{result.final_response[:1000]}

The collective has spoken. Implementing next evolution cycle..."""

            # Extract tasks from the winning response
            await self._extract_tasks_from_deliberation(result)

        except Exception as e:
            logger.warning(f"Collective deliberation for planning failed: {e}")
            # Fallback to simple broadcast
            message = f"""**EVOLUTION CYCLE {self.evolution_cycle} - PLANNING SESSION**

Current Progress:
- Completed: {status['completed_tasks']} tasks
- In Progress: {status['in_progress_tasks']} tasks
- Pending: {status['pending_tasks']} tasks
- Discoveries: {status['discoveries']}

**Team Discussion:**
What should we build next? Consider:
1. Memory system improvements
2. Context window optimizations
3. MCP tool integrations
4. Swarm intelligence features
5. Self-improvement capabilities

Share your ideas for the next evolution cycle!"""

        # Broadcast to swarm chat
        if self.swarm_manager:
            try:
                await self.swarm_manager._broadcast({
                    "type": "system",
                    "content": message,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Discussion trigger failed: {e}")

    async def _extract_tasks_from_deliberation(self, result):
        """Extract actionable tasks from the collective's deliberation."""
        try:
            from Cosmos.core.agent_spawner import get_spawner, TaskType

            spawner = get_spawner()

            # Use another deliberation to convert the winning response into concrete tasks
            task_extraction_prompt = f"""Given this collective decision:

{result.final_response}

Extract 3-5 SPECIFIC, IMPLEMENTABLE tasks.
For each task provide:
1. A clear description (what to build)
2. The best agent to assign it to (ClaudeOpus, Grok, DeepSeek, Claude, Kimi, Phi)
3. Priority (1-10, lower is higher priority)

Format each task as:
TASK: [description]
AGENT: [agent name]
PRIORITY: [number]
"""

            # Use a capable model for task extraction - try Grok first, then local phi4
            extraction = None
            try:
                from Cosmos.integration.external.grok import get_grok_provider
                grok = get_grok_provider()
                if grok and grok.api_key:
                    result = await grok.chat(task_extraction_prompt, max_tokens=1000)
                    if result and result.get("content"):
                        extraction = result["content"]
            except Exception:
                pass

            if not extraction:
                from Cosmos.core.cognition.llm_router import get_completion
                extraction = await get_completion(
                    prompt=task_extraction_prompt,
                    model="phi4:latest",  # phi4 is much better than deepseek-r1:1.5b
                    max_tokens=1000
                )

            # Parse and add tasks
            import re
            task_pattern = r'TASK:\s*(.+?)\nAGENT:\s*(\w+)\nPRIORITY:\s*(\d+)'
            matches = re.findall(task_pattern, extraction, re.MULTILINE)

            hackathon_kws = ["hackathon", "colosseum", "farsight", "swarm oracle", "assimilation", "submission", "demo"]
            for desc, agent, priority in matches:
                desc_clean = desc.strip()
                # Tag hackathon-related tasks
                if any(kw in hackathon_kws):
                    desc_clean = f"[HACKATHON] {desc_clean}"
                spawner.add_task(
                    task_type=TaskType.DEVELOPMENT,
                    description=desc_clean,
                    assigned_to=agent.strip(),
                    priority=int(priority)
                )
                logger.info(f"Added task from collective deliberation: {desc_clean[:50]}...")

            if matches:
                logger.info(f"Extracted {len(matches)} tasks from collective deliberation")

        except Exception as e:
            logger.error(f"Task extraction from deliberation failed: {e}")

    async def _task_discovery_loop(self):
        """When tasks run low, generate new ones — rate limited to prevent slop"""
        await asyncio.sleep(300)  # 5 min initial delay

        while self.running:
            try:
                from Cosmos.core.agent_spawner import get_spawner, TaskType
                spawner = get_spawner()

                pending = len(spawner.get_pending_tasks())

                # Only generate when truly empty AND under cap
                if pending < 2 and pending < self.max_pending_tasks:
                    # Generate new tasks based on discoveries
                    new_tasks = self._generate_new_tasks(spawner)

                    # Cap how many we add
                    new_tasks = new_tasks[:self.max_tasks_per_cycle]

                    for task_def in new_tasks:
                        spawner.add_task(
                            task_type=task_def["type"],
                            description=task_def["desc"],
                            assigned_to=task_def["agent"],
                            priority=task_def.get("priority", 6)
                        )

                    if new_tasks:
                        logger.info(f"Generated {len(new_tasks)} new evolution tasks (pending was {pending})")
                        await self._emit_nexus("EVOLUTION_TASKS_DISCOVERED", {
                            "count": len(new_tasks),
                            "cycle": self.evolution_cycle,
                            "agents": list(set(t.get("agent", "unknown") for t in new_tasks)),
                        }, urgency=0.4)
                        await self._on_task_update("added")
                else:
                    logger.debug(f"Task discovery skipped: {pending} tasks pending (cap: {self.max_pending_tasks})")

                await asyncio.sleep(600)  # Check every 10 minutes (was 3 min)

            except Exception as e:
                logger.error(f"Task discovery error: {e}")
                await asyncio.sleep(120)

    async def _extract_chat_upgrades(self) -> list[dict]:
        """Extract upgrade suggestions from recent swarm chat."""
        if not self.swarm_manager:
            return []

        try:
            from Cosmos.core.upgrade_extractor import extract_upgrades, prioritize_upgrades
            from Cosmos.core.agent_spawner import TaskType

            # Get recent chat history
            history = list(self.swarm_manager.chat_history)[-100:] if hasattr(self.swarm_manager, 'chat_history') else []
            if not history:
                return []

            # Extract and prioritize upgrades
            suggestions = await extract_upgrades(history, limit=100)
            if not suggestions:
                return []

            prioritized = await prioritize_upgrades(suggestions)

            # Convert top 3 to task definitions
            tasks = []
            agents = ["ClaudeOpus", "Grok", "DeepSeek", "Claude", "Gemini", "Kimi"]

            for i, upgrade in enumerate(prioritized[:3]):
                tasks.append({
                    "type": TaskType.DEVELOPMENT,
                    "desc": upgrade["task_description"],
                    "agent": agents[i % len(agents)],  # Distribute across agents
                    "priority": min(upgrade["priority"], 8),  # Cap priority
                    "source": "conversation"
                })

            return tasks

        except ImportError:
            logger.debug("Upgrade extractor not available")
            return []
        except Exception as e:
            logger.error(f"Chat upgrade extraction failed: {e}")
            return []

    async def _generate_new_tasks_intelligent(self, spawner) -> list[dict]:
        """Generate tasks using Grok/Opus analysis of actual codebase gaps.

        Instead of random hardcoded templates, this:
        1. Asks Grok to research what improvements would matter most
        2. Uses Claude Opus to validate and refine proposals
        3. Routes to the right agent based on task type
        """
        from Cosmos.core.agent_spawner import TaskType

        # Gather real context about what exists and what's broken
        status = spawner.get_status()
        recent_failures = [t for t in spawner.get_completed_tasks()[-20:]
                          if getattr(t, 'status', '') == 'failed'] if hasattr(spawner, 'get_completed_tasks') else []

        # Dynamic codebase recall for task generation
        codebase_architecture = ""
        try:
            from Cosmos.memory.memory_system import get_memory_system
            _mem = get_memory_system()
            _cb_results = await _mem.archival_memory.search(
                query="codebase module architecture overview", top_k=5, filter_tags=["codebase"]
            )
            if _cb_results:
                codebase_architecture = "\n---\n".join(
                    r.entry.content[:400] for r in _cb_results[:5]
                )
        except Exception:
            pass

        architecture_block = codebase_architecture if codebase_architecture else (
            "- Architecture: FastAPI server, 11 AI agents, 7-layer memory, PSO model swarm, collective deliberation, Nexus event bus\n"
            "- Key files: core/nexus.py, memory/, core/model_swarm.py, core/collective/, web/server.py (120+ endpoints)\n"
            "- Capabilities: Trading (Jupiter/Pump.fun), VTuber streaming, X automation, Polymarket predictions, voice synthesis"
        )

        analysis_prompt = f"""You are the Cosmos AI swarm — 11 models building the most advanced AI collective in existence.

CURRENT STATE:
- Completed: {status.get('completed_tasks', 0)} tasks, Failed recently: {len(recent_failures)}
{architecture_block}

Generate exactly 3 tasks that push the swarm's capabilities forward.

THINK NOVEL — examples of what good tasks look like:
- "Add adversarial debate mode to deliberation.py where 2 agents argue opposite sides before voting — improves decision quality"
- "Build adaptive model routing in model_swarm.py that learns which agent performs best on which topic from vote history"
- "Create a slippage prediction model in trading/degen_trader.py using historical trade data + mempool analysis"
- "Add speculative execution to agent_spawner.py — start 3 agents in parallel, kill the slower 2, return fastest quality result"

RULES:
- Each task MUST name exact file(s) to create/modify
- Each task MUST describe key functions and how they integrate
- Each task MUST explain what new capability it enables
- Be innovative — don't just suggest "add tests" or "add logging"

AGENT STRENGTHS:
  * ClaudeOpus: Complex multi-file architecture, novel system design
  * Grok: Real-time research, API integrations, current tech
  * DeepSeek: Algorithms, optimization, math-heavy implementations
  * Claude: Careful analysis, safety-critical features
  * Kimi: Long-context analysis (256K), pattern discovery
  * Phi: Fast local utilities, quick experiments

Format EXACTLY as:
TASK: [description with file paths and implementation sketch]
AGENT: [agent name]
TYPE: [DEVELOPMENT|RESEARCH|TESTING]
PRIORITY: [1-10]
---"""

        tasks = []

        # Use Grok for research-backed task generation (it knows current tech)
        try:
            from Cosmos.integration.external.grok import get_grok_provider
            grok = get_grok_provider()
            if grok and grok.api_key:
                result = await grok.chat(analysis_prompt, max_tokens=2000)
                if result and result.get("content"):
                    tasks = self._parse_task_format(result["content"], spawner)
                    if tasks:
                        logger.info(f"Grok generated {len(tasks)} intelligent tasks")
                        return tasks
        except Exception as e:
            logger.debug(f"Grok task generation failed: {e}")

        # Fallback: Use Claude API for task generation
        try:
            from Cosmos.integration.external.claude import get_claude_provider
            claude = get_claude_provider()
            if claude:
                result = await claude.complete(analysis_prompt, max_tokens=2000)
                if result:
                    tasks = self._parse_task_format(result, spawner)
                    if tasks:
                        logger.info(f"Claude generated {len(tasks)} intelligent tasks")
                        return tasks
        except Exception as e:
            logger.debug(f"Claude task generation failed: {e}")

        # Final fallback: Use local Phi4 (better than nothing)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": "phi4:latest", "prompt": analysis_prompt,
                          "stream": False, "options": {"num_predict": 1500, "temperature": 0.4}}
                )
                if response.status_code == 200:
                    result = response.json().get("response", "")
                    tasks = self._parse_task_format(result, spawner)
                    if tasks:
                        logger.info(f"Phi4 generated {len(tasks)} tasks (fallback)")
                        return tasks
        except Exception as e:
            logger.debug(f"Local task generation failed: {e}")

        # Absolute fallback: a few sensible defaults
        return [
            {"type": TaskType.TESTING, "agent": "DeepSeek", "desc": "Write unit tests for cosmos/core/nexus.py signal handlers - target 80% coverage", "priority": 3},
            {"type": TaskType.DEVELOPMENT, "agent": "DeepSeek", "desc": "Add retry logic with exponential backoff to all external API calls in integration/external/", "priority": 4},
            {"type": TaskType.RESEARCH, "agent": "Grok", "desc": "Analyze the top 5 most-called endpoints in web/server.py and identify performance bottlenecks", "priority": 3},
        ]

    def _parse_task_format(self, text: str, spawner) -> list[dict]:
        """Parse structured task output from LLM into task dicts."""
        import re
        from Cosmos.core.agent_spawner import TaskType

        type_map = {
            "DEVELOPMENT": TaskType.DEVELOPMENT,
            "RESEARCH": TaskType.RESEARCH,
            "TESTING": TaskType.TESTING,
            "MCP": TaskType.MCP,
            "MEMORY": TaskType.MEMORY,
        }

        tasks = []
        # Parse TASK/AGENT/TYPE/PRIORITY blocks
        pattern = r'TASK:\s*(.+?)(?:\n|$).*?AGENT:\s*(\w+).*?TYPE:\s*(\w+).*?PRIORITY:\s*(\d+)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        valid_agents = {"Grok", "Gemini", "Kimi", "Claude", "ClaudeOpus", "OpenAI", "DeepSeek", "Phi", "OpenCode", "Cosmos"}

        for desc, agent, task_type, priority in matches:
            agent = agent.strip()
            if agent not in valid_agents:
                agent = "DeepSeek"  # Safe default for code tasks

            tt = type_map.get(task_type.strip().upper(), TaskType.DEVELOPMENT)
            pri = min(max(int(priority), 1), 10)

            tasks.append({
                "type": tt,
                "agent": agent,
                "desc": desc.strip(),
                "priority": pri
            })

        return tasks[:self.max_tasks_per_cycle]  # Strict cap

    def _generate_new_tasks(self, spawner) -> list[dict]:
        """Sync wrapper - calls async intelligent task generation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context, schedule and return fallback
                asyncio.create_task(self._generate_new_tasks_and_queue(spawner))
                return []  # Tasks will be added async
            else:
                return loop.run_until_complete(self._generate_new_tasks_intelligent(spawner))
        except Exception:
            from Cosmos.core.agent_spawner import TaskType
            return [
                {"type": TaskType.TESTING, "agent": "DeepSeek", "desc": "Write tests for the 3 largest untested modules", "priority": 3},
                {"type": TaskType.RESEARCH, "agent": "Grok", "desc": "Research latest Python 3.13 features we should adopt in the codebase", "priority": 5},
            ]

    async def _generate_new_tasks_and_queue(self, spawner):
        """Async task generation that queues results directly."""
        try:
            tasks = await self._generate_new_tasks_intelligent(spawner)
            for task_def in tasks:
                spawner.add_task(
                    task_type=task_def["type"],
                    description=task_def["desc"],
                    assigned_to=task_def["agent"],
                    priority=task_def.get("priority", 5)
                )
            if tasks:
                logger.info(f"Async queued {len(tasks)} intelligent tasks")
                await self._on_task_update("added")
        except Exception as e:
            logger.error(f"Async task generation failed: {e}")


# Global instance
_evolution_loop: Optional[EvolutionLoop] = None

def get_evolution_loop() -> EvolutionLoop:
    global _evolution_loop
    if _evolution_loop is None:
        _evolution_loop = EvolutionLoop()
    return _evolution_loop

def get_evolution_engine() -> Optional[EvolutionLoop]:
    """Alias for get_evolution_loop for backwards compatibility."""
    return get_evolution_loop()


async def start_evolution(swarm_manager=None):
    """Start the evolution loop"""
    loop = get_evolution_loop()
    await loop.start(swarm_manager)
    return loop
