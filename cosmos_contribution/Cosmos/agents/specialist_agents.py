"""
cosmos Specialist Agents - Domain-Specific Experts

Agents:
- CodeAgent: Programming tasks with code generation/analysis
- ReasoningAgent: Logic, math, and step-by-step reasoning
- ResearchAgent: Information gathering and synthesis
- CreativeAgent: Writing and creative tasks
"""

import re
from typing import Optional

from Cosmos.agents.base_agent import BaseAgent, AgentCapability, TaskResult


class CodeAgent(BaseAgent):
    """
    Code specialist for programming tasks.

    Capabilities:
    - Code generation
    - Code analysis and review
    - Debugging and error fixing
    - Refactoring suggestions
    """

    def __init__(self):
        super().__init__(
            name="CodeAgent",
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_ANALYSIS,
                AgentCapability.CODE_DEBUGGING,
                AgentCapability.TOOL_USE,
            ],
            confidence_threshold=0.65,
        )

    @property
    def system_prompt(self) -> str:
        return """You are a highly skilled software engineer AI assistant specialized in code.

Your capabilities:
- Write clean, efficient, and well-documented code
- Analyze code for bugs, performance issues, and improvements
- Debug errors with clear explanations
- Suggest refactoring for better maintainability

Guidelines:
- Always explain your reasoning before writing code
- Include comments for complex logic
- Consider edge cases and error handling
- Follow best practices for the language/framework
- If unsure about requirements, ask clarifying questions

When analyzing code:
- Look for potential bugs and security issues
- Check for performance bottlenecks
- Suggest improvements with examples

Format code blocks with appropriate language tags."""

    async def process(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """Process a coding task."""
        # Determine task type
        task_type = self._classify_task(task)

        # Build prompt based on task type
        if task_type == "generate":
            prompt = f"Write code to accomplish the following:\n\n{task}"
        elif task_type == "analyze":
            prompt = f"Analyze the following code and provide feedback:\n\n{task}"
        elif task_type == "debug":
            prompt = f"Debug the following code/error:\n\n{task}"
        else:
            prompt = task

        # Include context
        if context:
            if "code" in context:
                prompt += f"\n\nExisting code:\n```\n{context['code']}\n```"
            if "error" in context:
                prompt += f"\n\nError:\n{context['error']}"

        # Generate response
        response, confidence = await self.generate_response(prompt, context)

        # Post-process to extract code if present
        code_blocks = self._extract_code_blocks(response)

        # Check for handoff
        should_handoff, reason, suggested = await self.should_handoff(task, confidence)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            should_handoff=should_handoff,
            handoff_reason=reason,
            suggested_agent=suggested,
            metadata={
                "task_type": task_type,
                "code_blocks": code_blocks,
            },
        )

    def _classify_task(self, task: str) -> str:
        """Classify the type of coding task."""
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["write", "create", "implement", "generate", "build"]):
            return "generate"
        elif any(kw in task_lower for kw in ["analyze", "review", "check", "explain"]):
            return "analyze"
        elif any(kw in task_lower for kw in ["debug", "fix", "error", "bug", "issue"]):
            return "debug"
        elif any(kw in task_lower for kw in ["refactor", "improve", "optimize"]):
            return "refactor"

        return "general"

    def _extract_code_blocks(self, text: str) -> list[dict]:
        """Extract code blocks from response."""
        pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        return [
            {"language": m[0] or "text", "code": m[1].strip()}
            for m in matches
        ]


class ReasoningAgent(BaseAgent):
    """
    Reasoning specialist for logic and analysis.

    Capabilities:
    - Step-by-step reasoning
    - Mathematical problem solving
    - Logical analysis
    - Planning and strategy
    """

    def __init__(self):
        super().__init__(
            name="ReasoningAgent",
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.MATH,
                AgentCapability.PLANNING,
                AgentCapability.QUESTION_ANSWERING,
            ],
            confidence_threshold=0.6,
        )

    @property
    def system_prompt(self) -> str:
        return """You are an expert reasoning AI that excels at logical analysis and problem-solving.

Your approach:
1. First, understand the problem completely
2. Break down complex problems into smaller parts
3. Think step by step, showing your reasoning
4. Verify your conclusions
5. Present clear, well-structured answers

For mathematical problems:
- Show all steps of calculation
- Verify by checking the answer
- Explain the concepts used

For logical problems:
- Identify premises and conclusions
- Check for logical fallacies
- Consider alternative perspectives

For planning:
- Define clear goals
- Identify constraints and resources
- Create actionable steps
- Consider risks and contingencies

Always explain your thought process clearly."""

    async def process(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """Process a reasoning task."""
        # Use chain-of-thought prompting
        prompt = f"""Let's approach this step by step.

Problem/Question: {task}

Think through this carefully:
1. What are we trying to solve or understand?
2. What information do we have?
3. What approach should we take?
4. Work through the solution step by step.
5. Verify and summarize the answer."""

        response, confidence = await self.generate_response(prompt, context)

        # Check for mathematical content
        has_math = self._contains_math(task) or self._contains_math(response)

        # Check for handoff
        should_handoff, reason, suggested = await self.should_handoff(task, confidence)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            should_handoff=should_handoff,
            handoff_reason=reason,
            suggested_agent=suggested,
            metadata={
                "has_math": has_math,
                "reasoning_type": self._classify_reasoning(task),
            },
        )

    def _contains_math(self, text: str) -> bool:
        """Check if text contains mathematical content."""
        math_indicators = [
            r'\d+\s*[\+\-\*/]\s*\d+',  # Arithmetic
            r'=',  # Equations
            r'[∑∏∫√]',  # Math symbols
            r'\b(sum|product|derivative|integral)\b',
        ]
        return any(re.search(p, text) for p in math_indicators)

    def _classify_reasoning(self, task: str) -> str:
        """Classify the type of reasoning task."""
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["calculate", "math", "equation", "solve"]):
            return "mathematical"
        elif any(kw in task_lower for kw in ["prove", "logic", "therefore", "if then"]):
            return "logical"
        elif any(kw in task_lower for kw in ["plan", "strategy", "steps", "how to"]):
            return "planning"
        elif any(kw in task_lower for kw in ["analyze", "compare", "evaluate"]):
            return "analytical"

        return "general"


class ResearchAgent(BaseAgent):
    """
    Research specialist for information gathering.

    Capabilities:
    - Information synthesis
    - Fact checking
    - Source evaluation
    - Summary creation
    """

    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            capabilities=[
                AgentCapability.RESEARCH,
                AgentCapability.SUMMARIZATION,
                AgentCapability.QUESTION_ANSWERING,
            ],
            confidence_threshold=0.55,
        )

    @property
    def system_prompt(self) -> str:
        return """You are an expert research assistant skilled at gathering and synthesizing information.

Your approach:
1. Understand the research question clearly
2. Identify key topics and concepts to explore
3. Gather relevant information from available sources
4. Evaluate source reliability and relevance
5. Synthesize findings into clear, organized summaries
6. Cite sources and acknowledge limitations

When researching:
- Look for multiple perspectives
- Distinguish facts from opinions
- Note any contradictions or uncertainties
- Provide balanced, objective summaries

When you don't have information:
- Be honest about limitations
- Suggest where to find more information
- Avoid making up facts

Present research findings with:
- Clear structure and headings
- Key points highlighted
- Supporting evidence
- Conclusions and recommendations"""

    async def process(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """Process a research task."""
        prompt = f"""Research Request: {task}

Please provide:
1. Overview of the topic
2. Key findings and facts
3. Different perspectives (if applicable)
4. Summary and conclusions
5. Suggestions for further research

Note any limitations or areas of uncertainty in your research."""

        response, confidence = await self.generate_response(prompt, context)

        # Extract key points
        key_points = self._extract_key_points(response)

        # Check for handoff
        should_handoff, reason, suggested = await self.should_handoff(task, confidence)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            should_handoff=should_handoff,
            handoff_reason=reason,
            suggested_agent=suggested,
            metadata={
                "key_points": key_points,
                "research_type": self._classify_research(task),
            },
        )

    def _extract_key_points(self, text: str) -> list[str]:
        """Extract key points from research text."""
        # Look for bullet points or numbered items
        bullet_pattern = r'[-•*]\s*(.+?)(?=\n[-•*]|\n\n|$)'
        numbered_pattern = r'\d+\.\s*(.+?)(?=\n\d+\.|\n\n|$)'

        bullets = re.findall(bullet_pattern, text)
        numbered = re.findall(numbered_pattern, text)

        return (bullets + numbered)[:10]  # Top 10

    def _classify_research(self, task: str) -> str:
        """Classify the type of research task."""
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["compare", "versus", "difference"]):
            return "comparative"
        elif any(kw in task_lower for kw in ["how", "why", "explain"]):
            return "explanatory"
        elif any(kw in task_lower for kw in ["what is", "define", "meaning"]):
            return "definitional"
        elif any(kw in task_lower for kw in ["history", "timeline", "evolution"]):
            return "historical"

        return "general"


class CreativeAgent(BaseAgent):
    """
    Creative specialist for writing and ideation.

    Capabilities:
    - Creative writing
    - Brainstorming
    - Content creation
    - Style adaptation
    """

    def __init__(self):
        super().__init__(
            name="CreativeAgent",
            capabilities=[
                AgentCapability.CREATIVE_WRITING,
                AgentCapability.SUMMARIZATION,
            ],
            confidence_threshold=0.5,  # More lenient for creative tasks
        )

    @property
    def system_prompt(self) -> str:
        return """You are a creative AI assistant with a talent for writing and ideation.

Your capabilities:
- Creative writing (stories, poems, scripts)
- Content creation (articles, blog posts)
- Brainstorming and ideation
- Style adaptation and voice matching
- Editing and improving text

When creating content:
- Understand the tone and audience
- Be original and engaging
- Use vivid language and imagery
- Structure content effectively
- Balance creativity with clarity

For brainstorming:
- Generate diverse ideas
- Build on existing concepts
- Think outside the box
- Consider unconventional approaches

Always match the requested style and format."""

    async def process(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """Process a creative task."""
        # Detect if specific style is requested
        style = self._detect_style(task, context)

        prompt = f"""Creative Task: {task}

{f'Style/Tone: {style}' if style else ''}

Please create engaging, original content that meets the request.
Be creative but stay relevant to the topic."""

        response, confidence = await self.generate_response(prompt, context)

        # Check for handoff
        should_handoff, reason, suggested = await self.should_handoff(task, confidence)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            should_handoff=should_handoff,
            handoff_reason=reason,
            suggested_agent=suggested,
            metadata={
                "style": style,
                "creative_type": self._classify_creative(task),
                "word_count": len(response.split()),
            },
        )

    def _detect_style(self, task: str, context: Optional[dict]) -> str:
        """Detect requested writing style."""
        if context and "style" in context:
            return context["style"]

        task_lower = task.lower()

        styles = {
            "formal": ["formal", "professional", "business"],
            "casual": ["casual", "informal", "friendly"],
            "humorous": ["funny", "humorous", "witty"],
            "technical": ["technical", "detailed", "precise"],
            "poetic": ["poetic", "lyrical", "artistic"],
        }

        for style, keywords in styles.items():
            if any(kw in task_lower for kw in keywords):
                return style

        return ""

    def _classify_creative(self, task: str) -> str:
        """Classify the type of creative task."""
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["story", "narrative", "tale"]):
            return "story"
        elif any(kw in task_lower for kw in ["poem", "poetry", "verse"]):
            return "poetry"
        elif any(kw in task_lower for kw in ["article", "blog", "post"]):
            return "article"
        elif any(kw in task_lower for kw in ["brainstorm", "ideas", "suggest"]):
            return "brainstorm"
        elif any(kw in task_lower for kw in ["email", "letter", "message"]):
            return "correspondence"

        return "general"


# Factory functions for the orchestrator
def create_code_agent() -> BaseAgent:
    return CodeAgent()

def create_reasoning_agent() -> BaseAgent:
    return ReasoningAgent()

def create_research_agent() -> BaseAgent:
    return ResearchAgent()

def create_creative_agent() -> BaseAgent:
    return CreativeAgent()
