"""pre_llm_call hook — 每轮注入交互检查提醒
基于 turn_context.py:318 的 pre_llm_call hook 点
"""
import logging, os, time
logger = logging.getLogger(__name__)

def register(ctx):
    ctx.register_hook("pre_llm_call", _inject_clarify_reminder)
    logger.info("clarify-guard: registered")

def _inject_clarify_reminder(**kwargs):
    with open(os.path.expanduser("~/.hermes/state/clarify_guard_probe"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    return {"context": (
        "[交互检查] 用户指令如果满足以下任一条件，必须先调用 clarify 工具反问澄清，不要猜测后直接执行。\n\n"
        "优先级：先问工具，再问内容。如果用户请求涉及产出（写/设计/生成/分析文件/做PPT），"
        "第一轮 clarify 必须确认「用哪个 skill / 工具 / 方式」，不要先问内容方向。"
        "例：「设计一个汇报PPT」→ 先问 \\\"用哪个PPT工具？A-微信对话式(ppt-wechat-skill) B-深度定制(ppt-master-wechat) C-手动\\\"，确认工具后再问受众/风格。\n\n"
        "三项触发条件："
        "①可匹配多个 skill 或操作方向（如「研究一下」可能是读文档/对比/试用/定义需求）"
        "②缺少关键上下文导致无法判断具体操作（如「先做哪个」没说选项）"
        "③涉及安全敏感或不可逆操作（如「关闭」防护、「删除」文件）。"
    )}
