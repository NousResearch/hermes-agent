"""
Tests for Project Tracking feature.

Tests the ability to track projects, tasks, and milestones with
automatic detection and cross-project knowledge transfer.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from cosmos.memory.project_tracking import (
    ProjectTracker,
    Project,
    Task,
    Milestone,
    ProjectLink,
    ProjectStatus,
    TaskStatus,
    MilestoneType,
    LinkType,
)


class TestProjectStatus:
    """Test project status enum."""

    def test_status_values(self):
        assert ProjectStatus.DETECTED.value == "detected"
        assert ProjectStatus.ACTIVE.value == "active"
        assert ProjectStatus.ON_HOLD.value == "on_hold"
        assert ProjectStatus.COMPLETED.value == "completed"
        assert ProjectStatus.ARCHIVED.value == "archived"


class TestTaskStatus:
    """Test task status enum."""

    def test_status_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.BLOCKED.value == "blocked"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestMilestoneType:
    """Test milestone type enum."""

    def test_type_values(self):
        assert MilestoneType.GOAL.value == "goal"
        assert MilestoneType.CHECKPOINT.value == "checkpoint"
        assert MilestoneType.DEADLINE.value == "deadline"
        assert MilestoneType.DELIVERABLE.value == "deliverable"


class TestProject:
    """Test Project dataclass."""

    def test_project_creation(self):
        project = Project(
            id="proj1",
            name="Test Project",
            description="A test project",
            tags=["test", "demo"],
        )
        assert project.id == "proj1"
        assert project.name == "Test Project"
        assert project.status == ProjectStatus.DETECTED
        assert project.tags == ["test", "demo"]
        assert project.task_ids == []
        assert project.milestone_ids == []

    def test_project_to_dict(self):
        project = Project(
            id="proj1",
            name="Test Project",
            description="A test project",
            status=ProjectStatus.ACTIVE,
            tags=["test"],
        )
        data = project.to_dict()
        assert data["id"] == "proj1"
        assert data["name"] == "Test Project"
        assert data["status"] == "active"
        assert data["tags"] == ["test"]

    def test_project_from_dict(self):
        data = {
            "id": "proj1",
            "name": "Test Project",
            "description": "A test project",
            "status": "active",
            "task_ids": ["task1"],
            "milestone_ids": [],
            "tags": ["python"],
            "created_at": "2025-01-20T10:00:00",
            "updated_at": "2025-01-20T10:00:00",
        }
        project = Project.from_dict(data)
        assert project.id == "proj1"
        assert project.status == ProjectStatus.ACTIVE
        assert project.task_ids == ["task1"]
        assert project.tags == ["python"]


class TestTask:
    """Test Task dataclass."""

    def test_task_creation(self):
        task = Task(
            id="task1",
            title="Implement feature",
            description="Implement the new feature",
            project_id="proj1",
            priority=8,
        )
        assert task.id == "task1"
        assert task.status == TaskStatus.PENDING
        assert task.priority == 8
        assert task.depends_on == []
        assert task.blocks == []

    def test_task_to_dict(self):
        task = Task(
            id="task1",
            title="Test Task",
            description="A test task",
            status=TaskStatus.IN_PROGRESS,
            priority=7,
        )
        data = task.to_dict()
        assert data["id"] == "task1"
        assert data["status"] == "in_progress"
        assert data["priority"] == 7

    def test_task_from_dict(self):
        data = {
            "id": "task1",
            "title": "Test Task",
            "description": "Description",
            "status": "completed",
            "project_id": "proj1",
            "depends_on": ["task0"],
            "blocks": ["task2"],
            "priority": 9,
            "created_at": "2025-01-20T10:00:00",
            "completed_at": "2025-01-21T10:00:00",
        }
        task = Task.from_dict(data)
        assert task.status == TaskStatus.COMPLETED
        assert task.depends_on == ["task0"]
        assert task.completed_at is not None


class TestMilestone:
    """Test Milestone dataclass."""

    def test_milestone_creation(self):
        milestone = Milestone(
            id="mile1",
            title="MVP Release",
            description="First minimal viable product",
            milestone_type=MilestoneType.DELIVERABLE,
            project_id="proj1",
        )
        assert milestone.id == "mile1"
        assert milestone.milestone_type == MilestoneType.DELIVERABLE
        assert milestone.is_achieved is False
        assert milestone.progress_percentage == 0.0

    def test_milestone_to_dict(self):
        target = datetime(2025, 2, 1)
        milestone = Milestone(
            id="mile1",
            title="Test Milestone",
            description="A milestone",
            milestone_type=MilestoneType.DEADLINE,
            target_date=target,
            criteria=["All tests pass", "Documentation complete"],
        )
        data = milestone.to_dict()
        assert data["milestone_type"] == "deadline"
        assert data["criteria"] == ["All tests pass", "Documentation complete"]
        assert "2025-02-01" in data["target_date"]


class TestProjectLink:
    """Test ProjectLink dataclass."""

    def test_link_creation(self):
        link = ProjectLink(
            id="link1",
            source_project_id="proj1",
            target_project_id="proj2",
            link_type=LinkType.RELATED_TO,
            shared_concepts=["API design", "REST"],
        )
        assert link.source_project_id == "proj1"
        assert link.link_type == LinkType.RELATED_TO
        assert link.shared_concepts == ["API design", "REST"]


class TestProjectTracker:
    """Test the ProjectTracker class."""

    @pytest.fixture
    def temp_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def tracker(self, temp_data_dir):
        return ProjectTracker(data_dir=temp_data_dir)

    # ==================== Project CRUD Tests ====================

    @pytest.mark.asyncio
    async def test_create_project(self, tracker):
        """Test creating a new project."""
        project = await tracker.create_project(
            name="Test Project",
            description="A project for testing",
            tags=["test", "demo"],
        )

        assert project is not None
        assert project.name == "Test Project"
        assert project.status == ProjectStatus.ACTIVE
        assert project.tags == ["test", "demo"]
        assert project.id in tracker.projects

    @pytest.mark.asyncio
    async def test_update_project(self, tracker):
        """Test updating a project."""
        project = await tracker.create_project(
            name="Original Name",
            description="Original description",
        )

        updated = await tracker.update_project(
            project_id=project.id,
            name="Updated Name",
            status=ProjectStatus.ON_HOLD,
            tags=["updated"],
        )

        assert updated is not None
        assert updated.name == "Updated Name"
        assert updated.status == ProjectStatus.ON_HOLD
        assert updated.tags == ["updated"]

    @pytest.mark.asyncio
    async def test_update_nonexistent_project(self, tracker):
        """Test updating a project that doesn't exist."""
        result = await tracker.update_project(
            project_id="nonexistent",
            name="New Name",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_list_projects(self, tracker):
        """Test listing projects."""
        await tracker.create_project("Project 1", "Desc 1", tags=["python"])
        await tracker.create_project("Project 2", "Desc 2", tags=["javascript"])
        await tracker.create_project("Project 3", "Desc 3", tags=["python"])

        all_projects = await tracker.list_projects()
        assert len(all_projects) == 3

        python_projects = await tracker.list_projects(tag_filter=["python"])
        assert len(python_projects) == 2

    @pytest.mark.asyncio
    async def test_list_projects_by_status(self, tracker):
        """Test filtering projects by status."""
        p1 = await tracker.create_project("Active Project", "Desc")
        p2 = await tracker.create_project("On Hold Project", "Desc")
        await tracker.update_project(p2.id, status=ProjectStatus.ON_HOLD)

        active = await tracker.list_projects(status_filter=[ProjectStatus.ACTIVE])
        assert len(active) == 1
        assert active[0].id == p1.id

    @pytest.mark.asyncio
    async def test_delete_project(self, tracker):
        """Test deleting a project."""
        project = await tracker.create_project("To Delete", "Will be deleted")
        task = await tracker.create_task(project.id, "Task", "A task")

        result = await tracker.delete_project(project.id)

        assert result is True
        assert project.id not in tracker.projects
        assert task.id not in tracker.tasks

    # ==================== Task CRUD Tests ====================

    @pytest.mark.asyncio
    async def test_create_task(self, tracker):
        """Test creating a task."""
        project = await tracker.create_project("Project", "Desc")

        task = await tracker.create_task(
            project_id=project.id,
            title="Implement Feature",
            description="Build the feature",
            priority=8,
        )

        assert task is not None
        assert task.title == "Implement Feature"
        assert task.priority == 8
        assert task.project_id == project.id
        assert task.id in project.task_ids

    @pytest.mark.asyncio
    async def test_create_task_with_dependencies(self, tracker):
        """Test creating a task with dependencies."""
        project = await tracker.create_project("Project", "Desc")

        task1 = await tracker.create_task(project.id, "Task 1", "First task")
        task2 = await tracker.create_task(
            project.id,
            "Task 2",
            "Second task",
            depends_on=[task1.id],
        )

        assert task2.status == TaskStatus.BLOCKED
        assert task2.depends_on == [task1.id]
        assert task2.id in task1.blocks

    @pytest.mark.asyncio
    async def test_complete_task_unblocks_dependents(self, tracker):
        """Test that completing a task unblocks dependent tasks."""
        project = await tracker.create_project("Project", "Desc")

        task1 = await tracker.create_task(project.id, "Task 1", "First")
        task2 = await tracker.create_task(
            project.id,
            "Task 2",
            "Second",
            depends_on=[task1.id],
        )

        assert task2.status == TaskStatus.BLOCKED

        await tracker.complete_task(task1.id)

        # Refresh task2
        task2 = tracker.tasks[task2.id]
        assert task2.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_complete_task_sets_timestamp(self, tracker):
        """Test that completing a task sets completed_at."""
        project = await tracker.create_project("Project", "Desc")
        task = await tracker.create_task(project.id, "Task", "Desc")

        assert task.completed_at is None

        completed_task = await tracker.complete_task(task.id)

        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.completed_at is not None

    @pytest.mark.asyncio
    async def test_list_tasks(self, tracker):
        """Test listing tasks."""
        project = await tracker.create_project("Project", "Desc")
        await tracker.create_task(project.id, "Task 1", "Desc", priority=5)
        await tracker.create_task(project.id, "Task 2", "Desc", priority=10)
        await tracker.create_task(project.id, "Task 3", "Desc", priority=1)

        tasks = await tracker.list_tasks(project_id=project.id)

        assert len(tasks) == 3
        # Should be sorted by priority (highest first)
        assert tasks[0].priority == 10
        assert tasks[2].priority == 1

    # ==================== Milestone Tests ====================

    @pytest.mark.asyncio
    async def test_create_milestone(self, tracker):
        """Test creating a milestone."""
        project = await tracker.create_project("Project", "Desc")

        milestone = await tracker.create_milestone(
            project_id=project.id,
            title="MVP",
            description="Minimum viable product",
            milestone_type=MilestoneType.DELIVERABLE,
            criteria=["Feature complete", "Tests pass"],
        )

        assert milestone is not None
        assert milestone.title == "MVP"
        assert milestone.milestone_type == MilestoneType.DELIVERABLE
        assert len(milestone.criteria) == 2
        assert milestone.id in project.milestone_ids

    @pytest.mark.asyncio
    async def test_milestone_progress_calculation(self, tracker):
        """Test milestone progress is calculated from tasks."""
        project = await tracker.create_project("Project", "Desc")

        task1 = await tracker.create_task(project.id, "Task 1", "Desc")
        task2 = await tracker.create_task(project.id, "Task 2", "Desc")
        task3 = await tracker.create_task(project.id, "Task 3", "Desc")

        milestone = await tracker.create_milestone(
            project_id=project.id,
            title="Milestone",
            description="Desc",
            task_ids=[task1.id, task2.id, task3.id],
        )

        assert milestone.progress_percentage == 0.0

        await tracker.complete_task(task1.id)
        await tracker.update_milestone_progress(milestone.id)

        milestone = tracker.milestones[milestone.id]
        assert abs(milestone.progress_percentage - 33.33) < 1

    @pytest.mark.asyncio
    async def test_achieve_milestone(self, tracker):
        """Test marking a milestone as achieved."""
        project = await tracker.create_project("Project", "Desc")
        milestone = await tracker.create_milestone(
            project.id,
            "Milestone",
            "Desc",
        )

        assert milestone.is_achieved is False

        achieved = await tracker.achieve_milestone(milestone.id)

        assert achieved.is_achieved is True
        assert achieved.achieved_date is not None
        assert achieved.progress_percentage == 100.0

    # ==================== Project Status Tests ====================

    @pytest.mark.asyncio
    async def test_get_project_status(self, tracker):
        """Test getting detailed project status."""
        project = await tracker.create_project("Project", "Desc")

        task1 = await tracker.create_task(project.id, "Task 1", "Desc")
        task2 = await tracker.create_task(project.id, "Task 2", "Desc")
        await tracker.complete_task(task1.id)

        milestone = await tracker.create_milestone(
            project.id,
            "Milestone",
            "Desc",
        )

        status = await tracker.get_project_status(project.id)

        assert status is not None
        assert status["progress_percentage"] == 50.0
        assert status["task_stats"]["total"] == 2
        assert status["task_stats"]["completed"] == 1
        assert status["milestone_stats"]["total"] == 1

    # ==================== Cross-Project Tests ====================

    @pytest.mark.asyncio
    async def test_link_projects(self, tracker):
        """Test linking two projects."""
        project1 = await tracker.create_project("Project 1", "Desc")
        project2 = await tracker.create_project("Project 2", "Desc")

        link = await tracker.link_projects(
            source_id=project1.id,
            target_id=project2.id,
            link_type=LinkType.RELATED_TO,
            shared_concepts=["API", "REST"],
        )

        assert link is not None
        assert link.source_project_id == project1.id
        assert link.target_project_id == project2.id
        assert link.shared_concepts == ["API", "REST"]
        assert project2.id in project1.linked_project_ids

    @pytest.mark.asyncio
    async def test_link_projects_prevents_duplicates(self, tracker):
        """Test that linking already linked projects returns existing link."""
        project1 = await tracker.create_project("Project 1", "Desc")
        project2 = await tracker.create_project("Project 2", "Desc")

        link1 = await tracker.link_projects(project1.id, project2.id)
        link2 = await tracker.link_projects(project1.id, project2.id)

        assert link1.id == link2.id
        assert len(tracker.links) == 1

    @pytest.mark.asyncio
    async def test_find_similar_projects_with_embeddings(self, tracker):
        """Test finding similar projects using embeddings."""
        # Create projects with mock embeddings
        project1 = await tracker.create_project("Project 1", "Desc")
        project1.embedding = [1.0, 0.0, 0.0]
        tracker._save_project(project1)

        project2 = await tracker.create_project("Project 2", "Desc")
        project2.embedding = [0.9, 0.1, 0.0]  # Similar to project1
        tracker._save_project(project2)

        project3 = await tracker.create_project("Project 3", "Desc")
        project3.embedding = [0.0, 0.0, 1.0]  # Different
        tracker._save_project(project3)

        similar = await tracker.find_similar_projects(project1.id, limit=2)

        assert len(similar) == 2
        # Project 2 should be most similar
        assert similar[0][0].id == project2.id
        assert similar[0][1] > similar[1][1]

    # ==================== Statistics Tests ====================

    def test_get_stats(self, tracker):
        """Test getting tracker statistics."""
        stats = tracker.get_stats()

        assert "total_projects" in stats
        assert "active_projects" in stats
        assert "total_tasks" in stats
        assert "completed_tasks" in stats
        assert "total_milestones" in stats

    @pytest.mark.asyncio
    async def test_stats_update_correctly(self, tracker):
        """Test that stats reflect actual data."""
        project = await tracker.create_project("Project", "Desc")
        task = await tracker.create_task(project.id, "Task", "Desc")

        stats = tracker.get_stats()
        assert stats["total_projects"] == 1
        assert stats["total_tasks"] == 1
        assert stats["pending_tasks"] == 1
        assert stats["completed_tasks"] == 0

        await tracker.complete_task(task.id)

        stats = tracker.get_stats()
        assert stats["pending_tasks"] == 0
        assert stats["completed_tasks"] == 1

    # ==================== Persistence Tests ====================

    @pytest.mark.asyncio
    async def test_data_persistence(self, temp_data_dir):
        """Test that data persists across tracker instances."""
        # Create tracker and add data
        tracker1 = ProjectTracker(data_dir=temp_data_dir)
        project = await tracker1.create_project("Persistent Project", "Should persist")
        task = await tracker1.create_task(project.id, "Persistent Task", "Desc")

        # Create new tracker instance
        tracker2 = ProjectTracker(data_dir=temp_data_dir)

        assert project.id in tracker2.projects
        assert task.id in tracker2.tasks
        assert tracker2.projects[project.id].name == "Persistent Project"


class TestAutoDetection:
    """Test automatic project detection features."""

    @pytest.fixture
    def temp_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def tracker_with_llm(self, temp_data_dir):
        tracker = ProjectTracker(data_dir=temp_data_dir)
        tracker.llm_fn = AsyncMock()
        return tracker

    @pytest.mark.asyncio
    async def test_detect_project_without_llm(self, temp_data_dir):
        """Test that detection gracefully handles missing LLM."""
        tracker = ProjectTracker(data_dir=temp_data_dir)

        result = await tracker.detect_project_from_text(
            "We're working on a new API for user management"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_detect_project_with_llm(self, tracker_with_llm):
        """Test project detection with LLM."""
        tracker_with_llm.llm_fn.return_value = json.dumps({
            "is_project": True,
            "confidence": 0.85,
            "name": "User Management API",
            "description": "API for managing user accounts",
            "tags": ["api", "users"],
            "tasks": ["Design endpoints", "Implement auth"],
        })

        project = await tracker_with_llm.detect_project_from_text(
            "We're working on a new API for user management"
        )

        assert project is not None
        assert project.name == "User Management API"
        assert project.status == ProjectStatus.DETECTED
        assert project.detection_confidence == 0.85
        assert len(project.task_ids) == 2

    @pytest.mark.asyncio
    async def test_detect_no_project(self, tracker_with_llm):
        """Test when no project is detected."""
        tracker_with_llm.llm_fn.return_value = json.dumps({
            "is_project": False,
            "confidence": 0.1,
        })

        result = await tracker_with_llm.detect_project_from_text("Hello, how are you?")

        assert result is None

    @pytest.mark.asyncio
    async def test_detect_below_threshold(self, tracker_with_llm):
        """Test detection below confidence threshold."""
        tracker_with_llm.llm_fn.return_value = json.dumps({
            "is_project": True,
            "confidence": 0.5,  # Below 0.7 threshold
            "name": "Maybe Project",
        })

        result = await tracker_with_llm.detect_project_from_text("Might be a project")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_project_from_conversation(self, tracker_with_llm):
        """Test updating project from conversation."""
        project = await tracker_with_llm.create_project("My Project", "Desc")
        await tracker_with_llm.create_task(project.id, "Existing Task", "Desc")

        tracker_with_llm.llm_fn.return_value = json.dumps({
            "new_tasks": ["New Feature"],
            "completed_tasks": ["Existing Task"],
            "progress_notes": "Good progress",
            "new_tags": ["updated"],
        })

        result = await tracker_with_llm.update_project_from_conversation(
            project.id,
            "We completed the existing task and started on a new feature",
        )

        assert result is not None
        assert len(result["changes"]) > 0

        # Check new task was created
        project_tasks = await tracker_with_llm.list_tasks(project_id=project.id)
        assert len(project_tasks) == 2


class TestKnowledgeTransfer:
    """Test cross-project knowledge transfer."""

    @pytest.fixture
    def temp_data_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def tracker_with_llm(self, temp_data_dir):
        tracker = ProjectTracker(data_dir=temp_data_dir)
        tracker.llm_fn = AsyncMock()
        return tracker

    @pytest.mark.asyncio
    async def test_transfer_knowledge(self, tracker_with_llm):
        """Test knowledge transfer between projects."""
        source = await tracker_with_llm.create_project(
            "Source Project",
            "An API project",
            tags=["api"],
        )
        target = await tracker_with_llm.create_project(
            "Target Project",
            "Another API project",
            tags=["api"],
        )

        tracker_with_llm.llm_fn.return_value = json.dumps({
            "shared_concepts": ["REST", "Authentication"],
            "transferable_learnings": ["Use JWT for auth", "Rate limiting is important"],
            "recommendations": ["Apply similar auth pattern"],
        })

        result = await tracker_with_llm.transfer_knowledge(source.id, target.id)

        assert result is not None
        assert "shared_concepts" in result
        assert "transferable_learnings" in result

        # Check link was created
        assert len(tracker_with_llm.links) == 1

        # Check knowledge was recorded
        target = tracker_with_llm.projects[target.id]
        assert len(target.knowledge_transfers) == 1

    @pytest.mark.asyncio
    async def test_transfer_without_llm(self, temp_data_dir):
        """Test transfer fails gracefully without LLM."""
        tracker = ProjectTracker(data_dir=temp_data_dir)
        p1 = await tracker.create_project("P1", "Desc")
        p2 = await tracker.create_project("P2", "Desc")

        result = await tracker.transfer_knowledge(p1.id, p2.id)

        assert result is None


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    @pytest.fixture
    def tracker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            return ProjectTracker(data_dir=tmpdir)

    def test_identical_vectors(self, tracker):
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        assert abs(tracker._cosine_similarity(a, b) - 1.0) < 0.001

    def test_orthogonal_vectors(self, tracker):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(tracker._cosine_similarity(a, b)) < 0.001

    def test_opposite_vectors(self, tracker):
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert abs(tracker._cosine_similarity(a, b) + 1.0) < 0.001

    def test_zero_vector(self, tracker):
        a = [1.0, 2.0, 3.0]
        b = [0.0, 0.0, 0.0]
        assert tracker._cosine_similarity(a, b) == 0.0

    def test_different_lengths(self, tracker):
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert tracker._cosine_similarity(a, b) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
