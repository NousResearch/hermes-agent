"""
Tests for Q2 2025 Advanced Agents features.

Tests cover:
- Planner Agent
- Critic Agent
- Web Agent
- File System Agent
- Agent Debates
- Specialization Learning
- Hierarchical Teams
"""

import pytest
import asyncio
from datetime import datetime


# ============ Planner Agent Tests ============

class TestPlannerAgent:
    """Tests for PlannerAgent task decomposition."""

    def test_planner_agent_import(self):
        """Test PlannerAgent can be imported."""
        from cosmos.agents.planner_agent import (
            PlannerAgent,
            Plan,
            SubTask,
            TaskStatus,
            TaskPriority,
        )
        assert PlannerAgent is not None

    @pytest.mark.asyncio
    async def test_create_plan_without_llm(self):
        """Test plan creation without LLM (single task fallback)."""
        from cosmos.agents.planner_agent import PlannerAgent, TaskStatus

        planner = PlannerAgent()
        plan = await planner.create_plan("Build a REST API")

        assert plan.id == "plan_1"
        assert plan.goal == "Build a REST API"
        assert len(plan.tasks) >= 1
        assert plan.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_add_task_to_plan(self):
        """Test adding tasks to existing plan."""
        from cosmos.agents.planner_agent import PlannerAgent

        planner = PlannerAgent()
        plan = await planner.create_plan("Test project")

        task = await planner.add_task(
            plan_id=plan.id,
            title="Write tests",
            description="Create unit tests",
            agent_type="code",
        )

        assert task.id in plan.tasks
        assert task.title == "Write tests"

    @pytest.mark.asyncio
    async def test_plan_execution(self):
        """Test plan execution flow."""
        from cosmos.agents.planner_agent import PlannerAgent, TaskStatus

        planner = PlannerAgent()
        plan = await planner.create_plan("Simple task")

        # Execute without agent executor
        completed_plan = await planner.execute_plan(plan.id)

        assert completed_plan.status == TaskStatus.COMPLETED
        assert completed_plan.completed_tasks >= 1

    def test_plan_progress_calculation(self):
        """Test plan progress percentage."""
        from cosmos.agents.planner_agent import Plan, SubTask, TaskStatus

        plan = Plan(id="test", goal="test goal")

        task1 = SubTask(id="t1", title="T1", description="D1", status=TaskStatus.COMPLETED)
        task2 = SubTask(id="t2", title="T2", description="D2", status=TaskStatus.PENDING)

        plan.tasks = {"t1": task1, "t2": task2}
        plan.total_tasks = 2
        plan.completed_tasks = 1

        assert plan.get_progress() == 0.5


# ============ Critic Agent Tests ============

class TestCriticAgent:
    """Tests for CriticAgent quality assurance."""

    def test_critic_agent_import(self):
        """Test CriticAgent can be imported."""
        from cosmos.agents.critic_agent import (
            CriticAgent,
            Review,
            QualityScore,
            QualityDimension,
            ReviewType,
        )
        assert CriticAgent is not None

    @pytest.mark.asyncio
    async def test_review_without_llm(self):
        """Test review with heuristic scoring."""
        from cosmos.agents.critic_agent import CriticAgent, ReviewType

        critic = CriticAgent()
        review = await critic.review(
            content="This is a sample text for review. It has multiple sentences.",
            review_type=ReviewType.TEXT,
        )

        assert review.id is not None
        assert 0 <= review.overall_score <= 1
        assert len(review.scores) > 0
        assert review.summary != ""

    @pytest.mark.asyncio
    async def test_compare_artifacts(self):
        """Test comparison of multiple artifacts."""
        from cosmos.agents.critic_agent import CriticAgent, ReviewType

        critic = CriticAgent()
        comparison = await critic.compare(
            artifacts=[
                "Short text",
                "This is a much longer and more detailed explanation with proper sentences.",
            ],
            review_type=ReviewType.TEXT,
        )

        assert "ranking" in comparison
        assert len(comparison["ranking"]) == 2
        assert "best" in comparison
        assert "worst" in comparison

    def test_quality_score_calculation(self):
        """Test overall score calculation."""
        from cosmos.agents.critic_agent import CriticAgent, QualityScore, QualityDimension

        critic = CriticAgent()

        scores = [
            QualityScore(
                dimension=QualityDimension.CORRECTNESS,
                score=0.8,
                confidence=0.9,
                feedback="Good",
            ),
            QualityScore(
                dimension=QualityDimension.CLARITY,
                score=0.6,
                confidence=0.8,
                feedback="OK",
            ),
        ]

        overall = critic._calculate_overall_score(scores)
        assert 0 <= overall <= 1


# ============ Web Agent Tests ============

class TestWebAgent:
    """Tests for WebAgent browsing capabilities."""

    def test_web_agent_import(self):
        """Test WebAgent can be imported."""
        from cosmos.agents.web_agent import (
            WebAgent,
            BrowsingSession,
            PageState,
            ActionType,
        )
        assert WebAgent is not None

    def test_page_classification(self):
        """Test page type classification."""
        from cosmos.agents.web_agent import WebAgent, PageState

        agent = WebAgent()

        # Login page
        state = PageState(url="https://example.com/login", title="Login", content="Enter username password")
        assert agent._classify_page(state) == "login"

        # Search page
        state = PageState(url="https://example.com/search?q=test", title="Search", content="Results")
        assert agent._classify_page(state) == "search"

    def test_session_creation(self):
        """Test browsing session structure."""
        from cosmos.agents.web_agent import BrowsingSession

        session = BrowsingSession(
            id="test_session",
            goal="Find information about Python",
        )

        assert session.id == "test_session"
        assert session.goal == "Find information about Python"
        assert len(session.actions) == 0


# ============ File System Agent Tests ============

class TestFileSystemAgent:
    """Tests for FileSystemAgent file operations."""

    def test_filesystem_agent_import(self):
        """Test FileSystemAgent can be imported."""
        from cosmos.agents.filesystem_agent import (
            FileSystemAgent,
            FileInfo,
            ProjectStructure,
            FileType,
        )
        assert FileSystemAgent is not None

    def test_project_type_detection(self):
        """Test project type detection."""
        from cosmos.agents.filesystem_agent import FileSystemAgent
        from pathlib import Path
        import tempfile
        import os

        agent = FileSystemAgent()

        # Create temp dir with pyproject.toml
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "pyproject.toml").write_text("[project]\nname='test'")
            project_type = agent._detect_project_type(Path(tmpdir))
            assert project_type == "python"

    def test_naming_convention_detection(self):
        """Test naming convention detection."""
        from cosmos.agents.filesystem_agent import FileSystemAgent
        from pathlib import Path
        import tempfile

        agent = FileSystemAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "snake_case_file.py").write_text("")
            (Path(tmpdir) / "another_snake.py").write_text("")

            convention = agent._detect_naming_convention(Path(tmpdir))
            assert convention == "snake_case"


# ============ Agent Debates Tests ============

class TestAgentDebates:
    """Tests for AgentDebates multi-agent discussion."""

    def test_debates_import(self):
        """Test AgentDebates can be imported."""
        from cosmos.agents.agent_debates import (
            AgentDebates,
            Debate,
            Argument,
            DebateRole,
        )
        assert AgentDebates is not None

    @pytest.mark.asyncio
    async def test_start_debate(self):
        """Test starting a debate."""
        from cosmos.agents.agent_debates import AgentDebates

        debates = AgentDebates()
        debate = await debates.start_debate(
            topic="Should we use microservices?",
            positions=["Yes, for scalability", "No, monolith is simpler"],
        )

        assert debate.id is not None
        assert debate.topic == "Should we use microservices?"
        assert len(debate.positions) == 2

    @pytest.mark.asyncio
    async def test_add_argument(self):
        """Test adding arguments to debate."""
        from cosmos.agents.agent_debates import AgentDebates, DebateRole

        debates = AgentDebates()
        debate = await debates.start_debate(topic="Test topic")

        arg = await debates.add_argument(
            debate_id=debate.id,
            agent_id="agent_1",
            content="This is my argument",
            role=DebateRole.PROPONENT,
            confidence=0.8,
        )

        assert arg.id in debate.arguments
        assert arg.confidence == 0.8

    def test_consensus_calculation(self):
        """Test consensus score calculation."""
        from cosmos.agents.agent_debates import (
            AgentDebates,
            Debate,
            Argument,
            DebateRole,
            ArgumentType,
        )

        debates = AgentDebates()
        debate = Debate(id="test", topic="test")

        # Add pro argument
        pro = Argument(
            id="a1",
            agent_id="agent1",
            role=DebateRole.PROPONENT,
            argument_type=ArgumentType.CLAIM,
            content="Pro argument",
            confidence=0.9,
        )
        pro.peer_ratings = {"agent2": 0.8}

        # Add con argument
        con = Argument(
            id="a2",
            agent_id="agent2",
            role=DebateRole.OPPONENT,
            argument_type=ArgumentType.REBUTTAL,
            content="Con argument",
            confidence=0.5,
        )
        con.peer_ratings = {"agent1": 0.6}

        debate.arguments = {"a1": pro, "a2": con}

        consensus = debates._calculate_consensus(debate)
        # Pro has higher weighted score, so should be positive
        assert consensus > 0


# ============ Specialization Learning Tests ============

class TestSpecializationLearning:
    """Tests for SpecializationLearning skill development."""

    def test_learning_import(self):
        """Test SpecializationLearning can be imported."""
        from cosmos.agents.specialization_learning import (
            SpecializationLearning,
            AgentProfile,
            Skill,
            SkillLevel,
        )
        assert SpecializationLearning is not None

    @pytest.mark.asyncio
    async def test_register_agent(self):
        """Test agent registration."""
        from cosmos.agents.specialization_learning import SpecializationLearning

        learning = SpecializationLearning()
        profile = await learning.register_agent(
            agent_id="agent_1",
            agent_type="code",
        )

        assert profile.agent_id == "agent_1"
        assert profile.agent_type == "code"
        assert "python" in profile.skills

    @pytest.mark.asyncio
    async def test_record_outcome(self):
        """Test recording task outcome."""
        from cosmos.agents.specialization_learning import SpecializationLearning

        learning = SpecializationLearning()
        await learning.register_agent("agent_1", "code")

        outcome = await learning.record_outcome(
            task_id="task_1",
            task_type="code",
            agent_id="agent_1",
            success=True,
            score=0.9,
        )

        assert outcome.success is True
        assert learning.agents["agent_1"].success_rate > 0

    @pytest.mark.asyncio
    async def test_get_best_agent(self):
        """Test best agent selection."""
        from cosmos.agents.specialization_learning import SpecializationLearning

        learning = SpecializationLearning()
        await learning.register_agent("code_agent", "code")
        await learning.register_agent("research_agent", "research")

        best = await learning.get_best_agent(
            task_type="code",
            available_agents=["code_agent", "research_agent"],
        )

        assert best == "code_agent"


# ============ Hierarchical Teams Tests ============

class TestHierarchicalTeams:
    """Tests for HierarchicalTeams coordination."""

    def test_teams_import(self):
        """Test HierarchicalTeams can be imported."""
        from cosmos.agents.hierarchical_teams import (
            HierarchicalTeams,
            Team,
            AgentNode,
            AgentRole,
        )
        assert HierarchicalTeams is not None

    @pytest.mark.asyncio
    async def test_create_agent_hierarchy(self):
        """Test creating agent hierarchy."""
        from cosmos.agents.hierarchical_teams import HierarchicalTeams, AgentRole

        teams = HierarchicalTeams()

        exec_agent = await teams.create_agent(
            agent_id="exec",
            name="Executive",
            role=AgentRole.EXECUTIVE,
        )

        manager = await teams.create_agent(
            agent_id="mgr1",
            name="Manager 1",
            role=AgentRole.MANAGER,
            manager_id="exec",
        )

        assert teams.executive_id == "exec"
        assert manager.manager_id == "exec"
        assert "mgr1" in teams.agents["exec"].subordinates

    @pytest.mark.asyncio
    async def test_form_team(self):
        """Test dynamic team formation."""
        from cosmos.agents.hierarchical_teams import (
            HierarchicalTeams,
            AgentRole,
            TeamStatus,
        )

        teams = HierarchicalTeams()

        # Create agents
        await teams.create_agent("mgr", "Manager", AgentRole.MANAGER, ["python"])
        await teams.create_agent("dev1", "Dev 1", AgentRole.SPECIALIST, ["python"])
        await teams.create_agent("dev2", "Dev 2", AgentRole.SPECIALIST, ["python"])

        team = await teams.form_team(
            purpose="Build feature X",
            required_specializations=["python"],
            team_size=2,
        )

        assert team.status == TeamStatus.ACTIVE
        assert len(team.member_ids) <= 2

    @pytest.mark.asyncio
    async def test_assign_task(self):
        """Test task assignment."""
        from cosmos.agents.hierarchical_teams import HierarchicalTeams, AgentRole

        teams = HierarchicalTeams()
        await teams.create_agent("spec1", "Specialist", AgentRole.SPECIALIST, ["code"])

        assignment = await teams.assign_task(
            task_type="code",
            description="Fix bug",
            preferred_agent="spec1",
        )

        assert assignment.assigned_to == "spec1"
        assert teams.agents["spec1"].current_tasks == 1

    def test_hierarchy_tree(self):
        """Test hierarchy tree generation."""
        from cosmos.agents.hierarchical_teams import HierarchicalTeams, AgentRole, AgentNode

        teams = HierarchicalTeams()

        exec_node = AgentNode(id="exec", name="Exec", role=AgentRole.EXECUTIVE)
        mgr_node = AgentNode(id="mgr", name="Mgr", role=AgentRole.MANAGER, manager_id="exec")
        exec_node.subordinates = ["mgr"]

        teams.agents = {"exec": exec_node, "mgr": mgr_node}
        teams.executive_id = "exec"

        tree = teams.get_hierarchy_tree()

        assert tree["id"] == "exec"
        assert len(tree["subordinates"]) == 1
        assert tree["subordinates"][0]["id"] == "mgr"


# ============ Integration Tests ============

class TestQ2Integration:
    """Integration tests for Q2 features."""

    @pytest.mark.asyncio
    async def test_planner_with_critic(self):
        """Test planner and critic working together."""
        from cosmos.agents.planner_agent import PlannerAgent
        from cosmos.agents.critic_agent import CriticAgent, ReviewType

        planner = PlannerAgent()
        critic = CriticAgent()

        # Create a plan
        plan = await planner.create_plan("Create a web app")

        # Review the plan
        plan_text = f"Goal: {plan.goal}\nTasks: {len(plan.tasks)}"
        review = await critic.review(plan_text, ReviewType.PLAN)

        assert review.overall_score > 0

    @pytest.mark.asyncio
    async def test_learning_with_teams(self):
        """Test specialization learning with hierarchical teams."""
        from cosmos.agents.specialization_learning import SpecializationLearning
        from cosmos.agents.hierarchical_teams import HierarchicalTeams, AgentRole

        learning = SpecializationLearning()
        teams = HierarchicalTeams()

        # Register agent in both systems
        await learning.register_agent("dev1", "code")
        await teams.create_agent("dev1", "Developer", AgentRole.SPECIALIST, ["code"])

        # Record outcome
        await learning.record_outcome(
            task_id="t1",
            task_type="code",
            agent_id="dev1",
            success=True,
            score=0.9,
        )

        # Both systems should have the agent
        assert "dev1" in learning.agents
        assert "dev1" in teams.agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
