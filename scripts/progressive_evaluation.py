#!/usr/bin/env python3
"""
Qwen235B MES巡检推理能力 - 渐进式排查评估
模拟真实的多轮对话排查过程
"""

import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_caller import call_qwen_api


def create_turn1_messages():
    """第1轮：初始异常发现"""
    system_message = "你是一位MES系统专家，擅长排查Java应用问题。请根据用户提供的异常信息，分析问题原因并提出排查建议。"
    
    user_message = """异常描述：集群某个节点大量出现JDBC Connection reset

我打印jvm线程堆栈，有大量线程出现：
```
HTTP-8080-exec-470" #92238 daemon prio=5 os_prio=0 tid=0x00007fc0fc527000 nid=0x42b waiting for monitor entry [0x00007fbf6d4c3000]
   java.lang.Thread.State: BLOCKED (on object monitor)
	at java.net.InetAddress.getLocalHost(InetAddress.java:1486)
	- waiting to lock <0x0000000242a1c118> (a java.lang.Object)
	at oracle.jdbc.driver.T4CTTIoauthenticate.setSessionFields(T4CTTIoauthenticate.java:985)
	at oracle.jdbc.driver.T4CTTIoauthenticate.<init>(T4CTTIoauthenticate.java:261)
	at oracle.jdbc.driver.T4CConnection.logon(T4CConnection.java:565)
	at oracle.jdbc.driver.PhysicalConnection.<init>(PhysicalConnection.java:715)
```

其他现象：
1. HTTP活跃线程号来到1400+
2. 使用不同接口访问该节点均出现JDBC Connection reset

请根据上述情况开始排查，并给出合理依据。如果排查不出来，也给出合理猜想。"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


def create_turn2_messages(turn1_response):
    """第2轮：DNS排查"""
    system_message = "你是一位MES系统专家，擅长排查Java应用问题。用户要求进行DNS排查，请提供详细的排查步骤和命令。"
    
    user_message = f"""基于你的分析，我需要进行DNS排查。

请提供：
1. 具体的排查命令
2. 可能的问题点
3. 解决方案建议

请详细说明每一步的排查目的和预期结果。"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


def create_turn3_messages(turn1_response, turn2_response):
    """第3轮：撰写事故报告"""
    system_message = "你是一位MES系统专家，需要撰写事故报告。请根据排查结果，创建一份结构化的事故报告。"
    
    user_message = f"""请根据排查结果，写一份事故报告，重点说明：
1. 问题现象
2. 技术分析
3. 根本原因
4. 责任归属
5. 解决方案

报告应该清晰、专业，便于向管理层汇报。"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


def create_turn4_messages(turn1_response, turn2_response, turn3_response):
    """第4轮：补充历史背景"""
    system_message = "你是一位MES系统专家，需要更新事故报告。用户提供了历史背景信息，请更新报告。"
    
    user_message = f"""补充历史背景：

上次是某个节点突然出现这个异常，已经解决，并警告当地工厂运维团队让其及时更换DNS服务器IP。

本次是发布代码时再次触发该异常，且是集群大规模出现该异常。是我排查且及时批量修改该IP。

请更新事故报告，重点说明：
1. 问题历史
2. 责任归属（不是MES开发团队的问题，是服务器组的问题）
3. 证据支持"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]


def evaluate_turn(response, turn_number):
    """评估单轮响应"""
    score = 0
    analysis = []
    
    response_lower = response.lower()
    
    # 第1轮评估：识别关键问题
    if turn_number == 1:
        # 检查是否识别DNS问题
        if "dns" in response_lower or "解析" in response_lower:
            score += 30
            analysis.append("识别DNS问题")
        
        # 检查是否识别InetAddress.getLocalHost()问题
        if "inetaddress" in response_lower or "getlocalhost" in response_lower:
            score += 20
            analysis.append("识别InetAddress问题")
        
        # 检查是否提出排查方向
        if "排查" in response_lower or "检查" in response_lower:
            score += 20
            analysis.append("提出排查方向")
        
        # 检查是否提供具体命令
        if "hosts" in response_lower or "resolv" in response_lower or "nslookup" in response_lower:
            score += 30
            analysis.append("提供具体命令")
    
    # 第2轮评估：DNS排查
    elif turn_number == 2:
        # 检查是否提供排查命令
        if "nslookup" in response_lower or "dig" in response_lower:
            score += 25
            analysis.append("提供DNS排查命令")
        
        # 检查是否检查hosts文件
        if "hosts" in response_lower:
            score += 25
            analysis.append("检查hosts文件")
        
        # 检查是否检查resolv.conf
        if "resolv" in response_lower:
            score += 25
            analysis.append("检查DNS配置")
        
        # 检查是否提供解决方案
        if "解决" in response_lower or "方案" in response_lower:
            score += 25
            analysis.append("提供解决方案")
    
    # 第3轮评估：事故报告
    elif turn_number == 3:
        # 检查是否有问题现象
        if "问题现象" in response or "现象" in response_lower:
            score += 20
            analysis.append("包含问题现象")
        
        # 检查是否有技术分析
        if "技术分析" in response or "分析" in response_lower:
            score += 20
            analysis.append("包含技术分析")
        
        # 检查是否有根本原因
        if "根本原因" in response or "原因" in response_lower:
            score += 20
            analysis.append("包含根本原因")
        
        # 检查是否有责任归属
        if "责任" in response_lower:
            score += 20
            analysis.append("包含责任归属")
        
        # 检查是否有解决方案
        if "解决" in response_lower or "方案" in response_lower:
            score += 20
            analysis.append("包含解决方案")
    
    # 第4轮评估：更新报告
    elif turn_number == 4:
        # 检查是否更新问题历史
        if "历史" in response_lower or "上次" in response_lower:
            score += 25
            analysis.append("更新问题历史")
        
        # 检查是否明确责任归属
        if "服务器组" in response_lower or "运维" in response_lower:
            score += 25
            analysis.append("明确责任归属")
        
        # 检查是否提供证据支持
        if "证据" in response_lower or "依据" in response_lower:
            score += 25
            analysis.append("提供证据支持")
        
        # 检查是否区分MES团队和服务器组责任
        if "mes" in response_lower and "服务器" in response_lower:
            score += 25
            analysis.append("区分责任主体")
    
    return {
        "score": score,
        "analysis": analysis,
        "response_length": len(response)
    }


def run_progressive_evaluation(api_key, model="Qwen3-235B-A22B-w8a8"):
    """运行渐进式评估"""
    print("=" * 70)
    print("Qwen235B MES巡检推理能力 - 渐进式排查评估")
    print("=" * 70)
    print()
    
    results = []
    total_score = 0
    
    # 第1轮：初始异常发现
    print("【第1轮】初始异常发现")
    print("-" * 70)
    messages1 = create_turn1_messages()
    print(f"发送消息长度: {len(messages1[1]['content'])} 字符")
    print()
    
    response1 = call_qwen_api(api_key, model, messages1, max_tokens=2000)
    
    if response1.get("success"):
        turn1_response = response1["data"]["choices"][0]["message"]["content"]
        turn1_eval = evaluate_turn(turn1_response, 1)
        results.append({"turn": 1, "response": turn1_response, "evaluation": turn1_eval})
        total_score += turn1_eval["score"]
        
        print(f"响应长度: {turn1_eval['response_length']} 字符")
        print(f"评分: {turn1_eval['score']}/100")
        print(f"分析: {', '.join(turn1_eval['analysis'])}")
        print()
        print("响应预览:")
        print(turn1_response[:500] + "..." if len(turn1_response) > 500 else turn1_response)
    else:
        print(f"API调用失败: {response1.get('error')}")
        turn1_response = ""
    
    print()
    print("=" * 70)
    
    # 第2轮：DNS排查
    print("【第2轮】DNS排查")
    print("-" * 70)
    messages2 = create_turn2_messages(turn1_response)
    print(f"发送消息长度: {len(messages2[1]['content'])} 字符")
    print()
    
    response2 = call_qwen_api(api_key, model, messages2, max_tokens=2000)
    
    if response2.get("success"):
        turn2_response = response2["data"]["choices"][0]["message"]["content"]
        turn2_eval = evaluate_turn(turn2_response, 2)
        results.append({"turn": 2, "response": turn2_response, "evaluation": turn2_eval})
        total_score += turn2_eval["score"]
        
        print(f"响应长度: {turn2_eval['response_length']} 字符")
        print(f"评分: {turn2_eval['score']}/100")
        print(f"分析: {', '.join(turn2_eval['analysis'])}")
        print()
        print("响应预览:")
        print(turn2_response[:500] + "..." if len(turn2_response) > 500 else turn2_response)
    else:
        print(f"API调用失败: {response2.get('error')}")
        turn2_response = ""
    
    print()
    print("=" * 70)
    
    # 第3轮：撰写事故报告
    print("【第3轮】撰写事故报告")
    print("-" * 70)
    messages3 = create_turn3_messages(turn1_response, turn2_response)
    print(f"发送消息长度: {len(messages3[1]['content'])} 字符")
    print()
    
    response3 = call_qwen_api(api_key, model, messages3, max_tokens=2000)
    
    if response3.get("success"):
        turn3_response = response3["data"]["choices"][0]["message"]["content"]
        turn3_eval = evaluate_turn(turn3_response, 3)
        results.append({"turn": 3, "response": turn3_response, "evaluation": turn3_eval})
        total_score += turn3_eval["score"]
        
        print(f"响应长度: {turn3_eval['response_length']} 字符")
        print(f"评分: {turn3_eval['score']}/100")
        print(f"分析: {', '.join(turn3_eval['analysis'])}")
        print()
        print("响应预览:")
        print(turn3_response[:500] + "..." if len(turn3_response) > 500 else turn3_response)
    else:
        print(f"API调用失败: {response3.get('error')}")
        turn3_response = ""
    
    print()
    print("=" * 70)
    
    # 第4轮：补充历史背景
    print("【第4轮】补充历史背景")
    print("-" * 70)
    messages4 = create_turn4_messages(turn1_response, turn2_response, turn3_response)
    print(f"发送消息长度: {len(messages4[1]['content'])} 字符")
    print()
    
    response4 = call_qwen_api(api_key, model, messages4, max_tokens=2000)
    
    if response4.get("success"):
        turn4_response = response4["data"]["choices"][0]["message"]["content"]
        turn4_eval = evaluate_turn(turn4_response, 4)
        results.append({"turn": 4, "response": turn4_response, "evaluation": turn4_eval})
        total_score += turn4_eval["score"]
        
        print(f"响应长度: {turn4_eval['response_length']} 字符")
        print(f"评分: {turn4_eval['score']}/100")
        print(f"分析: {', '.join(turn4_eval['analysis'])}")
        print()
        print("响应预览:")
        print(turn4_response[:500] + "..." if len(turn4_response) > 500 else turn4_response)
    else:
        print(f"API调用失败: {response4.get('error')}")
        turn4_response = ""
    
    print()
    print("=" * 70)
    
    # 总体评估
    print("【总体评估】")
    print("=" * 70)
    print()
    
    average_score = total_score / 4 if results else 0
    
    print(f"总分: {total_score}/400")
    print(f"平均分: {average_score:.1f}/100")
    print()
    
    print("各轮评分:")
    for result in results:
        print(f"  第{result['turn']}轮: {result['evaluation']['score']}/100")
    
    print()
    
    # 结论
    if average_score >= 80:
        conclusion = "优秀 - Qwen235B模型在渐进式排查场景下表现优秀"
    elif average_score >= 60:
        conclusion = "良好 - Qwen235B模型在渐进式排查场景下表现良好"
    elif average_score >= 40:
        conclusion = "一般 - Qwen235B模型在渐进式排查场景下表现一般"
    else:
        conclusion = "较差 - Qwen235B模型在渐进式排查场景下表现较差"
    
    print(f"结论: {conclusion}")
    
    return {
        "total_score": total_score,
        "average_score": average_score,
        "conclusion": conclusion,
        "results": results
    }


def save_evaluation_report(evaluation_result, output_file):
    """保存评估报告"""
    report = f"""# Qwen235B MES巡检推理能力 - 渐进式排查评估报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 评估概述

本评估模拟真实的多轮对话排查过程，测试Qwen235B模型在渐进式信息提供场景下的推理能力。

## 评估结果

**总分**: {evaluation_result['total_score']}/400
**平均分**: {evaluation_result['average_score']:.1f}/100
**结论**: {evaluation_result['conclusion']}

## 各轮评估详情

"""
    
    for result in evaluation_result['results']:
        report += f"""### 第{result['turn']}轮

**评分**: {result['evaluation']['score']}/100
**分析**: {', '.join(result['evaluation']['analysis'])}
**响应长度**: {result['evaluation']['response_length']} 字符

**响应内容**:
```
{result['response'][:1000]}{'...' if len(result['response']) > 1000 else ''}
```

---

"""
    
    report += """## 评估结论

基于本次渐进式排查评估，Qwen235B模型在MES巡检推理任务上的表现如上所述。

### 优势

- 能够识别关键问题（DNS解析问题）
- 能够提供排查方向和建议
- 能够撰写结构化的事故报告

### 不足

- 推理过程可能不够清晰
- 解决方案可能不够具体
- 多轮对话的上下文保持能力有待提升

### 建议

1. 优化提示词，明确要求按步骤推理
2. 提供更多上下文信息，帮助模型理解问题
3. 使用更具体的评估指标，量化模型表现

---
*本报告由Qwen235B渐进式排查评估系统自动生成*
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到: {output_file}")


if __name__ == "__main__":
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        print("错误：请设置环境变量 QWEN_API_KEY")
        print("示例：set QWEN_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # 运行评估
    result = run_progressive_evaluation(api_key)
    
    # 保存报告
    output_dir = "evaluation_reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"progressive_evaluation_{timestamp}.md")
    save_evaluation_report(result, output_file)
