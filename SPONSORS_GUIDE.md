# Eruditi 赞助与贡献指南

## 如何为 Eruditi 账号开启 GitHub Sponsors

### 开启步骤
1. **登录 GitHub 账号**
2. **访问 GitHub Sponsors 页面**：导航到 [https://github.com/sponsors](https://github.com/sponsors)
3. **点击 "Get sponsored"** 按钮
4. **填写个人资料**：
   - 完善个人/组织资料
   - 添加简介和头像
   - 设置赞助等级和权益
5. **设置付款方式**：
   - 绑定银行账户或 PayPal
   - 完成税务信息填写
6. **发布赞助页面**：审核通过后，赞助页面将正式上线

### 资格要求
- 必须是活跃的开源贡献者
- 拥有至少一个公开的仓库
- 遵守 GitHub 社区准则

## 赞助页面文案

### 主标题
**支持 Eruditi 的开源之旅**

### 简介
Eruditi 致力于构建高质量的开源项目，特别是在 AI 代理和工具链领域。您的赞助将帮助我们持续改进代码质量、添加新功能，并为社区提供更好的文档和支持。

### 赞助等级

| 等级 | 金额 | 权益 |
|------|------|------|
| 支持者 | $5/月 | - 专属感谢信息<br>- 项目更新通知<br>- 加入 Discord 社区 |
| 贡献者 | $20/月 | - 所有支持者权益<br>- 代码审查优先级<br>- 月度项目进度报告 |
| 合作伙伴 | $50/月 | - 所有贡献者权益<br>- 定制功能请求<br>- 季度视频会议<br>- 项目 README 鸣谢 |
| 企业赞助 | $200/月 | - 所有合作伙伴权益<br>- 企业 logo 展示<br>- 技术支持<br>- 定制解决方案 |

### 资金用途
- **开发与维护**：持续改进代码质量和功能
- **社区支持**：回答问题、解决 issue、提供文档
- **基础设施**：服务器、域名、开发工具
- **教育内容**：教程、示例、技术博客

### 为什么赞助 Eruditi
- **高质量代码**：我们注重代码质量和用户体验
- **开放透明**：所有资金使用情况公开透明
- **社区驱动**：重视社区反馈，持续改进
- **创新精神**：不断探索 AI 代理领域的新可能性

## 为 Hermes Agent 做更多贡献

### 贡献方式

1. **提交 PR**：
   - 修复 bug
   - 改进现有功能
   - 添加新工具或技能
   - 优化性能

2. **报告问题**：
   - 提交详细的 bug 报告
   - 提出新功能建议
   - 改进文档

3. **文档贡献**：
   - 编写教程
   - 完善 API 文档
   - 翻译文档

4. **测试**：
   - 测试新功能
   - 报告兼容性问题
   - 提供性能反馈

### 贡献指南
1. **Fork 仓库**：在 GitHub 上 fork [Hermes Agent](https://github.com/NousResearch/hermes-agent) 仓库
2. **克隆到本地**：`git clone https://github.com/your-username/hermes-agent.git`
3. **创建分支**：`git checkout -b feature/your-feature`
4. **安装依赖**：
   ```bash
   uv venv venv --python 3.11
   source venv/bin/activate
   uv pip install -e ".[all,dev]"
   ```
5. **编写代码**：遵循项目代码风格和最佳实践
6. **运行测试**：`pytest tests/ -v`
7. **提交代码**：使用 Conventional Commits 格式
8. **创建 PR**：在 GitHub 上提交 Pull Request

### 推荐贡献方向
- **工具扩展**：添加新的工具集成
- **技能开发**：创建有用的技能
- **平台支持**：添加新的消息平台集成
- **性能优化**：提高代码性能和可靠性
- **文档改进**：完善用户文档和开发者指南

## 适合贡献的热门项目

### 1. Hermes Agent
- **描述**：自改进 AI 代理，具有内置学习循环
- **技术栈**：Python, 多种 LLM API
- **贡献难度**：中等
- **链接**：[https://github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)

### 2. LangChain
- **描述**：构建 LLM 应用的框架
- **技术栈**：Python, TypeScript
- **贡献难度**：中等
- **链接**：[https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

### 3. LlamaIndex
- **描述**：数据增强的 LLM 应用框架
- **技术栈**：Python
- **贡献难度**：中等
- **链接**：[https://github.com/run-llama/llama_index](https://github.com/run-llama/llama_index)

### 4. Hugging Face Transformers
- **描述**：自然语言处理库
- **技术栈**：Python, PyTorch, TensorFlow
- **贡献难度**：较高
- **链接**：[https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

### 5. FastAPI
- **描述**：现代、快速的 Web API 框架
- **技术栈**：Python
- **贡献难度**：中等
- **链接**：[https://github.com/tiangolo/fastapi](https://github.com/tiangolo/fastapi)

### 6. PyTorch
- **描述**：深度学习框架
- **技术栈**：Python, C++
- **贡献难度**：较高
- **链接**：[https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)

### 7. NumPy
- **描述**：科学计算库
- **技术栈**：Python, C
- **贡献难度**：较高
- **链接**：[https://github.com/numpy/numpy](https://github.com/numpy/numpy)

### 8. pandas
- **描述**：数据分析库
- **技术栈**：Python, C
- **贡献难度**：中等
- **链接**：[https://github.com/pandas-dev/pandas](https://github.com/pandas-dev/pandas)

### 9. scikit-learn
- **描述**：机器学习库
- **技术栈**：Python, C
- **贡献难度**：中等
- **链接**：[https://github.com/scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn)

### 10. Jupyter
- **描述**：交互式计算环境
- **技术栈**：Python, JavaScript
- **贡献难度**：中等
- **链接**：[https://github.com/jupyter/jupyter](https://github.com/jupyter/jupyter)

## 贡献小贴士

1. **从简单开始**：先从修复小 bug 或改进文档开始
2. **阅读贡献指南**：每个项目都有自己的贡献指南
3. **加入社区**：参与项目的 Discord 或 Slack 社区
4. **提问**：如果有疑问，不要犹豫提问
5. **持续学习**：了解项目的代码结构和设计理念
6. **尊重维护者**：理解维护者的时间和精力有限
7. **保持耐心**：PR 可能需要时间审核
8. **庆祝成功**：每一个合并的 PR 都是对开源的贡献

## 联系我们

- **GitHub**：[https://github.com/Eruditi](https://github.com/Eruditi)
- **Discord**：加入我们的社区讨论
- **Email**：contact@eruditi.dev

感谢您对开源事业的支持！