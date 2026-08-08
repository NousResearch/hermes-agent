# tests/test_integration.py
"""End-to-end test: messages flow through channels correctly."""
import pytest
from server.channel import ChannelManager


@pytest.mark.asyncio
async def test_full_message_flow():
    mgr = ChannelManager()
    channels = ["#tasks", "#review", "#consensus", "#general"]
    for ch in channels:
        await mgr.create_channel(ch)

    tasks_received = []
    reviews_received = []
    consensus_received = []
    general_received = []

    async def task_handler(msg):
        tasks_received.append(msg)

    async def review_handler(msg):
        reviews_received.append(msg)

    async def consensus_handler(msg):
        consensus_received.append(msg)

    async def general_handler(msg):
        general_received.append(msg)

    await mgr.subscribe("#tasks", task_handler)
    await mgr.subscribe("#review", review_handler)
    await mgr.subscribe("#consensus", consensus_handler)
    await mgr.subscribe("#general", general_handler)

    # Simulate the full AI2050 workflow
    await mgr.publish("#tasks", {
        "channel": "#tasks", "sender": "supervisor",
        "content": "任务: T001 - 验证频谱分工振荡定理",
        "msg_type": "task",
        "metadata": {"task_id": "T001", "title": "频谱分工振荡定理"}
    })

    await mgr.publish("#review", {
        "channel": "#review", "sender": "deepseek-researcher",
        "content": "研究完成。关键发现：Band3(β)与语法解析耦合最强(c=0.78)",
        "msg_type": "review",
        "metadata": {"task_id": "T001"}
    })

    await mgr.publish("#consensus", {
        "channel": "#consensus", "sender": "gpt-reviewer",
        "content": "GPT-5.4 评审: 方法论正确，样本量需扩大...",
        "msg_type": "review_analysis",
        "metadata": {"task_id": "T001", "reviewer_role": "reviewer+synthesizer"}
    })

    await mgr.publish("#consensus", {
        "channel": "#consensus", "sender": "claude-reviewer",
        "content": "Claude 4.6 评审: 数学严谨性良好，需补充时域验证...",
        "msg_type": "review_analysis",
        "metadata": {"task_id": "T001", "reviewer_role": "mathematical-rigor"}
    })

    await mgr.publish("#general", {
        "channel": "#general", "sender": "gpt-reviewer",
        "content": "综合共识: ACCEPTED. 需补充 [1] 扩大样本 [2] 时域验证",
        "msg_type": "consensus",
        "metadata": {"task_id": "T001"}
    })

    await mgr.publish("#general", {
        "channel": "#general", "sender": "deepseek-researcher",
        "content": "DeepSeek总结: R定律新增验证数据，brain_grounding+0.03",
        "msg_type": "researcher_summary",
        "metadata": {"task_id": "T001"}
    })

    await mgr.drain()

    assert len(tasks_received) == 1
    assert tasks_received[0]["msg_type"] == "task"
    assert tasks_received[0]["metadata"]["task_id"] == "T001"

    assert len(reviews_received) == 1
    assert reviews_received[0]["msg_type"] == "review"
    assert "Band3" in reviews_received[0]["content"]

    assert len(consensus_received) == 2
    assert consensus_received[0]["sender"] == "gpt-reviewer"
    assert consensus_received[1]["sender"] == "claude-reviewer"

    assert len(general_received) == 2
    assert general_received[0]["msg_type"] == "consensus"
    assert general_received[1]["msg_type"] == "researcher_summary"
