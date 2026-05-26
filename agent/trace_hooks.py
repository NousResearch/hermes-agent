"""
Hermes Agent 轨迹钩子 - Rust 后端版本

自动记录工具调用到 Rust 评估中心，用于训练和评估。
"""

import json
import time
import uuid
from functools import wraps
from typing import Any, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)

# 全局开关（从配置读取）
TRACING_ENABLED = True

# 评估中心连接（延迟初始化）
_eval_center = None


def get_eval_center():
    """获取评估中心实例（延迟初始化）"""
    global _eval_center
    if _eval_center is None:
        try:
            # 尝试导入 Rust 评估中心
            import hermes_eval_center
            
            # 使用 Hermes home 目录
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()
            db_path = os.path.join(hermes_home, "eval_center.db")
            
            _eval_center = hermes_eval_center.PyEvalCenter(db_path)
            logger.info(f"✅ 轨迹钩子已启用（Rust 后端）: {db_path}")
        except ImportError as e:
            logger.warning(f"⚠️ Rust 评估中心不可用，回退到内存存储: {e}")
            _eval_center = InMemoryTraceStore()
        except Exception as e:
            logger.warning(f"⚠️ 无法初始化评估中心: {e}")
            _eval_center = InMemoryTraceStore()
    return _eval_center


class InMemoryTraceStore:
    """内存轨迹存储（回退实现）"""
    
    def __init__(self):
        self.traces = []
        self.active_traces = {}
    
    def submit_trace(self, trace_id: str, task: str, input_data: str, 
                    output: str, tool_calls: str):
        """提交轨迹"""
        self.traces.append({
            'trace_id': trace_id,
            'task': task,
            'input': input_data,
            'output': output,
            'tool_calls': tool_calls,
            'timestamp': time.time(),
        })
        
        if len(self.traces) % 10 == 0:
            logger.info(f"📊 已记录 {len(self.traces)} 条轨迹（内存）")
    
    def get_state(self, trace_id: str) -> str:
        """获取状态"""
        return "candidate"
    
    def score_trace(self, trace_id: str) -> float:
        """评分"""
        return 0.5
    
    def get_stats(self) -> str:
        """获取统计"""
        return json.dumps({
            'total': len(self.traces),
            'backend': 'memory'
        })
    
    def export_traces(self, state: str, limit: int) -> str:
        """导出轨迹"""
        return json.dumps(self.traces[:limit])


def trace_tool_call(func):
    """
    装饰器：自动记录工具调用到评估中心
    
    用法：
        @trace_tool_call
        def handle_function_call(function_name, function_args, task_id=None):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TRACING_ENABLED:
            return func(*args, **kwargs)
        
        # 生成轨迹 ID
        trace_id = str(uuid.uuid4())
        
        # 提取参数
        function_name = args[0] if len(args) > 0 else kwargs.get('function_name', 'unknown')
        function_args = args[1] if len(args) > 1 else kwargs.get('function_args', {})
        task_id = kwargs.get('task_id', None)
        
        # 开始时间
        start_time = time.time()
        success = False
        result = None
        error = None
        
        try:
            # 调用原函数
            result = func(*args, **kwargs)
            success = True
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            # 结束时间
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # 记录到评估中心
            try:
                eval_center = get_eval_center()
                
                # 准备数据
                task = f"{function_name}"
                input_data = json.dumps(function_args) if isinstance(function_args, dict) else str(function_args)
                output = str(result)[:1000] if result else ""
                tool_calls = json.dumps([{
                    'function': function_name,
                    'args': function_args,
                    'duration_ms': duration_ms,
                    'success': success,
                }])
                
                # 提交轨迹
                eval_center.submit_trace(
                    trace_id,
                    task,
                    input_data,
                    output,
                    tool_calls
                )
                
                # 自动评分
                if success:
                    score = eval_center.score_trace(trace_id)
                    if score >= 0.6:
                        logger.debug(f"✅ 轨迹 {trace_id[:8]} 评分通过: {score:.2f}")
                
            except Exception as e:
                logger.debug(f"⚠️ 记录轨迹失败: {e}")
    
    return wrapper


def get_trace_stats() -> Dict[str, Any]:
    """获取轨迹统计信息"""
    try:
        eval_center = get_eval_center()
        stats_json = eval_center.get_stats()
        return json.loads(stats_json)
    except Exception as e:
        logger.error(f"获取统计失败: {e}")
        return {'total': 0, 'error': str(e)}


def export_traces(output_path: str, state: str = "active", limit: int = 100):
    """导出轨迹到文件"""
    try:
        eval_center = get_eval_center()
        traces_json = eval_center.export_traces(state, limit)
        
        with open(output_path, 'w') as f:
            f.write(traces_json)
        
        traces = json.loads(traces_json)
        logger.info(f"✅ 已导出 {len(traces)} 条轨迹到 {output_path}")
        return len(traces)
    except Exception as e:
        logger.error(f"导出轨迹失败: {e}")
        return 0


# 向后兼容的别名
def get_eval_center_stats():
    """向后兼容：获取评估中心统计"""
    return get_trace_stats()
