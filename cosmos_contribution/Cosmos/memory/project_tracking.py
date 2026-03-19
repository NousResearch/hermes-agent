"""
cosmos Project Tracking - Track projects with automatic detection and cross-project knowledge transfer

Features:
- Automatic project detection from conversations
- Task and milestone management with dependencies
- Progress tracking with completion metrics
- Cross-project knowledge transfer and linking
- Semantic similarity for finding related projects
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from enum import Enum

from loguru import logger


class ProjectStatus(Enum):
    """Status of a project."""
    DETECTED = "detected"      # Auto-detected from conversation
    ACTIVE = "active"          # Actively being worked on
    ON_HOLD = "on_hold"        # Paused
    COMPLETED = "completed"    # Finished
    ARCHIVED = "archived"      # No longer relevant


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class MilestoneType(Enum):
    """Type of milestone."""
    GOAL = "goal"              # High-level objective
    CHECKPOINT = "checkpoint"  # Progress marker
    DEADLINE = "deadline"      # Time-bound target
    DELIVERABLE = "deliverable"  # Concrete output


class LinkType(Enum):
    """Type of project relationship."""
    DEPENDS_ON = "depends_on"        # This project depends on another
    RELATED_TO = "related_to"        # Projects share concepts/domain
    SUCCESSOR_OF = "successor_of"    # This project continues another
    INFORMS = "informs"              # Knowledge flows to another project


@dataclass
class Project:
    """A tracked project."""
    id: str
    name: str
    description: str
    status: ProjectStatus = ProjectStatus.DETECTED

    # Related items
    task_ids: list[str] = field(default_factory=list)
    milestone_ids: list[str] = field(default_factory=list)

    # Auto-detection metadata
    detection_confidence: float = 0.0
    detection_source: Optional[str] = None  # Text that triggered detection

    # Cross-project linking
    linked_project_ids: list[str] = field(default_factory=list)
    knowledge_transfers: list[dict] = field(default_factory=list)

    # Semantic data
    embedding: Optional[list[float]] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "task_ids": self.task_ids,
            "milestone_ids": self.milestone_ids,
            "detection_confidence": self.detection_confidence,
            "detection_source": self.detection_source,
            "linked_project_ids": self.linked_project_ids,
            "knowledge_transfers": self.knowledge_transfers,
            "embedding": self.embedding,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            status=ProjectStatus(data.get("status", "detected")),
            task_ids=data.get("task_ids", []),
            milestone_ids=data.get("milestone_ids", []),
            detection_confidence=data.get("detection_confidence", 0.0),
            detection_source=data.get("detection_source"),
            linked_project_ids=data.get("linked_project_ids", []),
            knowledge_transfers=data.get("knowledge_transfers", []),
            embedding=data.get("embedding"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
        )


@dataclass
class Task:
    """A task within a project."""
    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    project_id: Optional[str] = None

    # Dependencies
    depends_on: list[str] = field(default_factory=list)  # Task IDs this task depends on
    blocks: list[str] = field(default_factory=list)       # Task IDs blocked by this task

    # Priority (0-10, higher is more important)
    priority: int = 5

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    due_date: Optional[datetime] = None

    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "project_id": self.project_id,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            status=TaskStatus(data.get("status", "pending")),
            project_id=data.get("project_id"),
            depends_on=data.get("depends_on", []),
            blocks=data.get("blocks", []),
            priority=data.get("priority", 5),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            due_date=datetime.fromisoformat(data["due_date"]) if data.get("due_date") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Milestone:
    """A milestone within a project."""
    id: str
    title: str
    description: str
    milestone_type: MilestoneType = MilestoneType.CHECKPOINT
    project_id: Optional[str] = None

    # Progress tracking
    target_date: Optional[datetime] = None
    achieved_date: Optional[datetime] = None
    is_achieved: bool = False

    # Criteria for completion
    criteria: list[str] = field(default_factory=list)
    progress_percentage: float = 0.0

    # Related tasks
    task_ids: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)

    # Metadata
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "milestone_type": self.milestone_type.value,
            "project_id": self.project_id,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "achieved_date": self.achieved_date.isoformat() if self.achieved_date else None,
            "is_achieved": self.is_achieved,
            "criteria": self.criteria,
            "progress_percentage": self.progress_percentage,
            "task_ids": self.task_ids,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Milestone":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            milestone_type=MilestoneType(data.get("milestone_type", "checkpoint")),
            project_id=data.get("project_id"),
            target_date=datetime.fromisoformat(data["target_date"]) if data.get("target_date") else None,
            achieved_date=datetime.fromisoformat(data["achieved_date"]) if data.get("achieved_date") else None,
            is_achieved=data.get("is_achieved", False),
            criteria=data.get("criteria", []),
            progress_percentage=data.get("progress_percentage", 0.0),
            task_ids=data.get("task_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProjectLink:
    """A relationship between two projects."""
    id: str
    source_project_id: str
    target_project_id: str
    link_type: LinkType

    # Knowledge transfer details
    shared_concepts: list[str] = field(default_factory=list)
    transferred_learnings: list[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_project_id": self.source_project_id,
            "target_project_id": self.target_project_id,
            "link_type": self.link_type.value,
            "shared_concepts": self.shared_concepts,
            "transferred_learnings": self.transferred_learnings,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectLink":
        return cls(
            id=data["id"],
            source_project_id=data["source_project_id"],
            target_project_id=data["target_project_id"],
            link_type=LinkType(data.get("link_type", "related_to")),
            shared_concepts=data.get("shared_concepts", []),
            transferred_learnings=data.get("transferred_learnings", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            metadata=data.get("metadata", {}),
        )


class ProjectTracker:
    """
    Track projects with automatic detection, progress tracking, and cross-project knowledge transfer.

    Features:
    - CRUD operations for projects, tasks, and milestones
    - Automatic project detection from conversation text
    - Progress calculation based on task/milestone completion
    - Cross-project linking and knowledge transfer
    - Semantic similarity search for related projects
    """

    def __init__(
        self,
        data_dir: str = "./data",
        detection_threshold: float = 0.7,
    ):
        self.data_dir = Path(data_dir)
        self.detection_threshold = detection_threshold

        # In-memory storage
        self.projects: dict[str, Project] = {}
        self.tasks: dict[str, Task] = {}
        self.milestones: dict[str, Milestone] = {}
        self.links: dict[str, ProjectLink] = {}

        # Callbacks for memory integration
        self.get_entities_fn: Optional[Callable] = None
        self.embed_fn: Optional[Callable] = None
        self.llm_fn: Optional[Callable] = None

        # Create storage directories
        self._projects_dir = self.data_dir / "projects" / "projects"
        self._tasks_dir = self.data_dir / "projects" / "tasks"
        self._milestones_dir = self.data_dir / "projects" / "milestones"
        self._links_dir = self.data_dir / "projects" / "links"

        for dir_path in [self._projects_dir, self._tasks_dir, self._milestones_dir, self._links_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Load existing data
        self._load_all()

    def _load_all(self):
        """Load all data from disk."""
        # Load projects
        for file_path in self._projects_dir.glob("proj_*.json"):
            try:
                data = json.loads(file_path.read_text(encoding='utf-8'))
                project = Project.from_dict(data)
                self.projects[project.id] = project
            except Exception as e:
                logger.warning(f"Failed to load project from {file_path}: {e}")

        # Load tasks
        for file_path in self._tasks_dir.glob("task_*.json"):
            try:
                data = json.loads(file_path.read_text(encoding='utf-8'))
                task = Task.from_dict(data)
                self.tasks[task.id] = task
            except Exception as e:
                logger.warning(f"Failed to load task from {file_path}: {e}")

        # Load milestones
        for file_path in self._milestones_dir.glob("mile_*.json"):
            try:
                data = json.loads(file_path.read_text(encoding='utf-8'))
                milestone = Milestone.from_dict(data)
                self.milestones[milestone.id] = milestone
            except Exception as e:
                logger.warning(f"Failed to load milestone from {file_path}: {e}")

        # Load links
        for file_path in self._links_dir.glob("link_*.json"):
            try:
                data = json.loads(file_path.read_text(encoding='utf-8'))
                link = ProjectLink.from_dict(data)
                self.links[link.id] = link
            except Exception as e:
                logger.warning(f"Failed to load link from {file_path}: {e}")

        logger.info(f"Loaded {len(self.projects)} projects, {len(self.tasks)} tasks, "
                   f"{len(self.milestones)} milestones, {len(self.links)} links")

    def _save_project(self, project: Project):
        """Save a project to disk."""
        file_path = self._projects_dir / f"proj_{project.id}.json"
        file_path.write_text(json.dumps(project.to_dict(), indent=2), encoding='utf-8')

    def _save_task(self, task: Task):
        """Save a task to disk."""
        file_path = self._tasks_dir / f"task_{task.id}.json"
        file_path.write_text(json.dumps(task.to_dict(), indent=2), encoding='utf-8')

    def _save_milestone(self, milestone: Milestone):
        """Save a milestone to disk."""
        file_path = self._milestones_dir / f"mile_{milestone.id}.json"
        file_path.write_text(json.dumps(milestone.to_dict(), indent=2), encoding='utf-8')

    def _save_link(self, link: ProjectLink):
        """Save a link to disk."""
        file_path = self._links_dir / f"link_{link.id}.json"
        file_path.write_text(json.dumps(link.to_dict(), indent=2), encoding='utf-8')

    def _delete_file(self, directory: Path, prefix: str, item_id: str):
        """Delete a file from disk."""
        file_path = directory / f"{prefix}_{item_id}.json"
        if file_path.exists():
            file_path.unlink()

    # ==================== Project CRUD ====================

    async def create_project(
        self,
        name: str,
        description: str,
        tags: Optional[list[str]] = None,
        status: ProjectStatus = ProjectStatus.ACTIVE,
        metadata: Optional[dict] = None,
    ) -> Project:
        """
        Create a new project.

        Args:
            name: Project name
            description: Project description
            tags: Optional list of tags
            status: Initial project status
            metadata: Optional metadata dict

        Returns:
            The created Project
        """
        project_id = str(uuid.uuid4())[:8]

        project = Project(
            id=project_id,
            name=name,
            description=description,
            status=status,
            tags=tags or [],
            metadata=metadata or {},
        )

        # Generate embedding if available
        if self.embed_fn:
            try:
                project.embedding = await self.embed_fn(f"{name} {description}")
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")

        self.projects[project_id] = project
        self._save_project(project)

        logger.info(f"Created project: {name} (ID: {project_id})")
        return project

    async def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[ProjectStatus] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Project]:
        """
        Update an existing project.

        Args:
            project_id: Project to update
            name: New name (optional)
            description: New description (optional)
            status: New status (optional)
            tags: New tags (optional, replaces existing)
            metadata: Metadata to merge (optional)

        Returns:
            Updated Project or None if not found
        """
        project = self.projects.get(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return None

        if name is not None:
            project.name = name
        if description is not None:
            project.description = description
        if status is not None:
            project.status = status
        if tags is not None:
            project.tags = tags
        if metadata is not None:
            project.metadata.update(metadata)

        project.updated_at = datetime.now()

        # Update embedding if name or description changed
        if (name or description) and self.embed_fn:
            try:
                project.embedding = await self.embed_fn(f"{project.name} {project.description}")
            except Exception as e:
                logger.warning(f"Failed to update embedding: {e}")

        self._save_project(project)

        logger.info(f"Updated project: {project.name} (ID: {project_id})")
        return project

    async def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        return self.projects.get(project_id)

    async def list_projects(
        self,
        status_filter: Optional[list[ProjectStatus]] = None,
        tag_filter: Optional[list[str]] = None,
    ) -> list[Project]:
        """
        List projects with optional filters.

        Args:
            status_filter: Only include projects with these statuses
            tag_filter: Only include projects with any of these tags

        Returns:
            List of matching projects
        """
        results = []

        for project in self.projects.values():
            # Status filter
            if status_filter and project.status not in status_filter:
                continue

            # Tag filter
            if tag_filter and not any(t in project.tags for t in tag_filter):
                continue

            results.append(project)

        # Sort by updated_at (newest first)
        results.sort(key=lambda p: p.updated_at, reverse=True)
        return results

    async def delete_project(self, project_id: str) -> bool:
        """Delete a project and all associated tasks/milestones."""
        project = self.projects.get(project_id)
        if not project:
            return False

        # Delete associated tasks
        for task_id in project.task_ids[:]:
            await self.delete_task(task_id)

        # Delete associated milestones
        for milestone_id in project.milestone_ids[:]:
            await self.delete_milestone(milestone_id)

        # Delete project links
        links_to_delete = [
            link_id for link_id, link in self.links.items()
            if link.source_project_id == project_id or link.target_project_id == project_id
        ]
        for link_id in links_to_delete:
            del self.links[link_id]
            self._delete_file(self._links_dir, "link", link_id)

        # Delete project
        del self.projects[project_id]
        self._delete_file(self._projects_dir, "proj", project_id)

        logger.info(f"Deleted project: {project_id}")
        return True

    # ==================== Task CRUD ====================

    async def create_task(
        self,
        project_id: str,
        title: str,
        description: str,
        priority: int = 5,
        depends_on: Optional[list[str]] = None,
        due_date: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Task]:
        """
        Create a new task in a project.

        Args:
            project_id: Project to add the task to
            title: Task title
            description: Task description
            priority: Priority (0-10, higher is more important)
            depends_on: List of task IDs this task depends on
            due_date: Optional due date
            metadata: Optional metadata

        Returns:
            Created Task or None if project not found
        """
        project = self.projects.get(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return None

        task_id = str(uuid.uuid4())[:8]

        task = Task(
            id=task_id,
            title=title,
            description=description,
            project_id=project_id,
            priority=max(0, min(10, priority)),
            depends_on=depends_on or [],
            due_date=due_date,
            metadata=metadata or {},
        )

        # Set initial status based on dependencies
        if depends_on:
            # Check if any dependencies are not completed
            for dep_id in depends_on:
                dep_task = self.tasks.get(dep_id)
                if dep_task and dep_task.status != TaskStatus.COMPLETED:
                    task.status = TaskStatus.BLOCKED
                    break

        # Update dependency tracking
        for dep_id in task.depends_on:
            dep_task = self.tasks.get(dep_id)
            if dep_task and task_id not in dep_task.blocks:
                dep_task.blocks.append(task_id)
                self._save_task(dep_task)

        self.tasks[task_id] = task
        project.task_ids.append(task_id)
        project.updated_at = datetime.now()

        self._save_task(task)
        self._save_project(project)

        logger.info(f"Created task: {title} in project {project_id}")
        return task

    async def update_task(
        self,
        task_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        priority: Optional[int] = None,
        due_date: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Task]:
        """Update an existing task."""
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Task not found: {task_id}")
            return None

        if title is not None:
            task.title = title
        if description is not None:
            task.description = description
        if status is not None:
            task.status = status
            if status == TaskStatus.COMPLETED:
                task.completed_at = datetime.now()
        if priority is not None:
            task.priority = max(0, min(10, priority))
        if due_date is not None:
            task.due_date = due_date
        if metadata is not None:
            task.metadata.update(metadata)

        self._save_task(task)

        # Update project timestamp
        if task.project_id:
            project = self.projects.get(task.project_id)
            if project:
                project.updated_at = datetime.now()
                self._save_project(project)

        logger.info(f"Updated task: {task.title}")
        return task

    async def complete_task(self, task_id: str) -> Optional[Task]:
        """
        Mark a task as completed and unblock dependent tasks.

        Returns:
            Updated Task or None if not found
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Task not found: {task_id}")
            return None

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        self._save_task(task)

        # Unblock dependent tasks
        for blocked_id in task.blocks:
            blocked_task = self.tasks.get(blocked_id)
            if blocked_task and blocked_task.status == TaskStatus.BLOCKED:
                # Check if all dependencies are now completed
                all_deps_complete = all(
                    self.tasks.get(dep_id) and self.tasks.get(dep_id).status == TaskStatus.COMPLETED
                    for dep_id in blocked_task.depends_on
                )
                if all_deps_complete:
                    blocked_task.status = TaskStatus.PENDING
                    self._save_task(blocked_task)
                    logger.info(f"Unblocked task: {blocked_task.title}")

        # Update project timestamp
        if task.project_id:
            project = self.projects.get(task.project_id)
            if project:
                project.updated_at = datetime.now()
                self._save_project(project)

        logger.info(f"Completed task: {task.title}")
        return task

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    async def list_tasks(
        self,
        project_id: Optional[str] = None,
        status_filter: Optional[list[TaskStatus]] = None,
    ) -> list[Task]:
        """List tasks with optional filters."""
        results = []

        for task in self.tasks.values():
            if project_id and task.project_id != project_id:
                continue
            if status_filter and task.status not in status_filter:
                continue
            results.append(task)

        # Sort by priority (highest first), then by created_at
        results.sort(key=lambda t: (-t.priority, t.created_at))
        return results

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        task = self.tasks.get(task_id)
        if not task:
            return False

        # Remove from project
        if task.project_id:
            project = self.projects.get(task.project_id)
            if project and task_id in project.task_ids:
                project.task_ids.remove(task_id)
                project.updated_at = datetime.now()
                self._save_project(project)

        # Remove from milestones
        for milestone in self.milestones.values():
            if task_id in milestone.task_ids:
                milestone.task_ids.remove(task_id)
                self._save_milestone(milestone)

        # Update dependencies
        for other_task in self.tasks.values():
            if task_id in other_task.depends_on:
                other_task.depends_on.remove(task_id)
                self._save_task(other_task)
            if task_id in other_task.blocks:
                other_task.blocks.remove(task_id)
                self._save_task(other_task)

        del self.tasks[task_id]
        self._delete_file(self._tasks_dir, "task", task_id)

        logger.info(f"Deleted task: {task_id}")
        return True

    # ==================== Milestone CRUD ====================

    async def create_milestone(
        self,
        project_id: str,
        title: str,
        description: str,
        milestone_type: MilestoneType = MilestoneType.CHECKPOINT,
        target_date: Optional[datetime] = None,
        criteria: Optional[list[str]] = None,
        task_ids: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Milestone]:
        """
        Create a new milestone in a project.

        Args:
            project_id: Project to add the milestone to
            title: Milestone title
            description: Milestone description
            milestone_type: Type of milestone
            target_date: Target completion date
            criteria: List of completion criteria
            task_ids: Tasks that contribute to this milestone
            metadata: Optional metadata

        Returns:
            Created Milestone or None if project not found
        """
        project = self.projects.get(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return None

        milestone_id = str(uuid.uuid4())[:8]

        milestone = Milestone(
            id=milestone_id,
            title=title,
            description=description,
            milestone_type=milestone_type,
            project_id=project_id,
            target_date=target_date,
            criteria=criteria or [],
            task_ids=task_ids or [],
            metadata=metadata or {},
        )

        # Calculate initial progress
        if milestone.task_ids:
            milestone.progress_percentage = await self._calculate_milestone_progress(milestone)

        self.milestones[milestone_id] = milestone
        project.milestone_ids.append(milestone_id)
        project.updated_at = datetime.now()

        self._save_milestone(milestone)
        self._save_project(project)

        logger.info(f"Created milestone: {title} in project {project_id}")
        return milestone

    async def _calculate_milestone_progress(self, milestone: Milestone) -> float:
        """Calculate milestone progress based on task completion."""
        if not milestone.task_ids:
            return 0.0

        completed = sum(
            1 for task_id in milestone.task_ids
            if self.tasks.get(task_id) and self.tasks.get(task_id).status == TaskStatus.COMPLETED
        )
        return (completed / len(milestone.task_ids)) * 100.0

    async def achieve_milestone(self, milestone_id: str) -> Optional[Milestone]:
        """
        Mark a milestone as achieved.

        Returns:
            Updated Milestone or None if not found
        """
        milestone = self.milestones.get(milestone_id)
        if not milestone:
            logger.warning(f"Milestone not found: {milestone_id}")
            return None

        milestone.is_achieved = True
        milestone.achieved_date = datetime.now()
        milestone.progress_percentage = 100.0
        self._save_milestone(milestone)

        # Update project timestamp
        if milestone.project_id:
            project = self.projects.get(milestone.project_id)
            if project:
                project.updated_at = datetime.now()
                self._save_project(project)

        logger.info(f"Achieved milestone: {milestone.title}")
        return milestone

    async def update_milestone_progress(self, milestone_id: str) -> Optional[Milestone]:
        """Recalculate and update milestone progress."""
        milestone = self.milestones.get(milestone_id)
        if not milestone:
            return None

        milestone.progress_percentage = await self._calculate_milestone_progress(milestone)
        self._save_milestone(milestone)
        return milestone

    async def get_milestone(self, milestone_id: str) -> Optional[Milestone]:
        """Get a milestone by ID."""
        return self.milestones.get(milestone_id)

    async def list_milestones(
        self,
        project_id: Optional[str] = None,
        achieved_only: bool = False,
    ) -> list[Milestone]:
        """List milestones with optional filters."""
        results = []

        for milestone in self.milestones.values():
            if project_id and milestone.project_id != project_id:
                continue
            if achieved_only and not milestone.is_achieved:
                continue
            results.append(milestone)

        # Sort by target_date (soonest first), then by created_at
        results.sort(key=lambda m: (m.target_date or datetime.max, m.created_at))
        return results

    async def delete_milestone(self, milestone_id: str) -> bool:
        """Delete a milestone."""
        milestone = self.milestones.get(milestone_id)
        if not milestone:
            return False

        # Remove from project
        if milestone.project_id:
            project = self.projects.get(milestone.project_id)
            if project and milestone_id in project.milestone_ids:
                project.milestone_ids.remove(milestone_id)
                project.updated_at = datetime.now()
                self._save_project(project)

        del self.milestones[milestone_id]
        self._delete_file(self._milestones_dir, "mile", milestone_id)

        logger.info(f"Deleted milestone: {milestone_id}")
        return True

    # ==================== Auto-Detection ====================

    async def detect_project_from_text(self, text: str) -> Optional[Project]:
        """
        Automatically detect a project from conversation text using LLM.

        Args:
            text: Conversation text to analyze

        Returns:
            Detected Project or None if no project detected
        """
        if not self.llm_fn:
            logger.warning("LLM function not configured for project detection")
            return None

        try:
            prompt = f"""Analyze the following text and determine if it describes a project.
If it does, extract the project information.

Text:
{text}

Respond in JSON format:
{{
    "is_project": true/false,
    "confidence": 0.0-1.0,
    "name": "project name",
    "description": "brief description",
    "tags": ["tag1", "tag2"],
    "tasks": ["task1", "task2"]
}}

If is_project is false, other fields can be empty."""

            response = await self.llm_fn(prompt)

            # Parse response
            if isinstance(response, str):
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return None
            else:
                data = response

            if not data.get("is_project", False):
                return None

            confidence = data.get("confidence", 0.0)
            if confidence < self.detection_threshold:
                logger.info(f"Project detection confidence ({confidence}) below threshold ({self.detection_threshold})")
                return None

            # Check for existing similar project
            existing = await self._find_similar_by_name(data.get("name", ""))
            if existing:
                logger.info(f"Found existing similar project: {existing.name}")
                return existing

            # Create new project
            project = await self.create_project(
                name=data.get("name", "Untitled Project"),
                description=data.get("description", ""),
                tags=data.get("tags", []),
                status=ProjectStatus.DETECTED,
            )

            project.detection_confidence = confidence
            project.detection_source = text[:500]  # Store first 500 chars
            self._save_project(project)

            # Create initial tasks if detected
            for task_title in data.get("tasks", []):
                await self.create_task(
                    project_id=project.id,
                    title=task_title,
                    description=f"Auto-detected task: {task_title}",
                )

            logger.info(f"Auto-detected project: {project.name} (confidence: {confidence})")
            return project

        except Exception as e:
            logger.error(f"Project detection failed: {e}")
            return None

    async def _find_similar_by_name(self, name: str) -> Optional[Project]:
        """Find a project with a similar name."""
        name_lower = name.lower()
        for project in self.projects.values():
            if project.name.lower() == name_lower:
                return project
            # Simple similarity check
            if name_lower in project.name.lower() or project.name.lower() in name_lower:
                return project
        return None

    async def update_project_from_conversation(
        self,
        project_id: str,
        text: str,
    ) -> Optional[dict]:
        """
        Update a project based on new conversation text.

        Uses LLM to extract updates, new tasks, or progress information.

        Returns:
            Dict with updates made or None if project not found
        """
        project = self.projects.get(project_id)
        if not project:
            logger.warning(f"Project not found: {project_id}")
            return None

        if not self.llm_fn:
            logger.warning("LLM function not configured")
            return None

        try:
            prompt = f"""Analyze this conversation in the context of the project "{project.name}".
Extract any updates, new tasks, or progress information.

Project description: {project.description}
Current tasks: {[t.title for tid in project.task_ids if (t := self.tasks.get(tid))]}

Conversation:
{text}

Respond in JSON format:
{{
    "new_tasks": ["task1", "task2"],
    "completed_tasks": ["task title to mark complete"],
    "progress_notes": "any progress update",
    "new_tags": ["additional tags"]
}}"""

            response = await self.llm_fn(prompt)

            # Parse response
            if isinstance(response, str):
                import re
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return None
            else:
                data = response

            updates = {"project_id": project_id, "changes": []}

            # Add new tasks
            for task_title in data.get("new_tasks", []):
                task = await self.create_task(
                    project_id=project_id,
                    title=task_title,
                    description=f"Added from conversation",
                )
                if task:
                    updates["changes"].append(f"Added task: {task_title}")

            # Mark tasks complete
            for task_title in data.get("completed_tasks", []):
                for task in self.tasks.values():
                    if task.project_id == project_id and task.title.lower() == task_title.lower():
                        await self.complete_task(task.id)
                        updates["changes"].append(f"Completed task: {task_title}")
                        break

            # Add new tags
            for tag in data.get("new_tags", []):
                if tag not in project.tags:
                    project.tags.append(tag)
                    updates["changes"].append(f"Added tag: {tag}")

            if updates["changes"]:
                project.updated_at = datetime.now()
                self._save_project(project)

            return updates

        except Exception as e:
            logger.error(f"Project update from conversation failed: {e}")
            return None

    # ==================== Progress Tracking ====================

    async def get_project_status(self, project_id: str) -> Optional[dict]:
        """
        Get detailed project status with progress metrics.

        Returns:
            Dict with project status information
        """
        project = self.projects.get(project_id)
        if not project:
            return None

        # Get tasks
        project_tasks = [
            self.tasks[tid] for tid in project.task_ids
            if tid in self.tasks
        ]

        # Calculate task stats
        task_stats = {
            "total": len(project_tasks),
            "pending": len([t for t in project_tasks if t.status == TaskStatus.PENDING]),
            "in_progress": len([t for t in project_tasks if t.status == TaskStatus.IN_PROGRESS]),
            "completed": len([t for t in project_tasks if t.status == TaskStatus.COMPLETED]),
            "blocked": len([t for t in project_tasks if t.status == TaskStatus.BLOCKED]),
            "cancelled": len([t for t in project_tasks if t.status == TaskStatus.CANCELLED]),
        }

        # Calculate overall progress
        if task_stats["total"] > 0:
            progress = (task_stats["completed"] / task_stats["total"]) * 100
        else:
            progress = 0.0

        # Get milestones
        project_milestones = [
            self.milestones[mid] for mid in project.milestone_ids
            if mid in self.milestones
        ]

        milestone_stats = {
            "total": len(project_milestones),
            "achieved": len([m for m in project_milestones if m.is_achieved]),
            "upcoming": len([
                m for m in project_milestones
                if not m.is_achieved and m.target_date and m.target_date > datetime.now()
            ]),
            "overdue": len([
                m for m in project_milestones
                if not m.is_achieved and m.target_date and m.target_date < datetime.now()
            ]),
        }

        # Get linked projects
        linked = []
        for link in self.links.values():
            if link.source_project_id == project_id:
                target = self.projects.get(link.target_project_id)
                if target:
                    linked.append({
                        "project_id": target.id,
                        "name": target.name,
                        "link_type": link.link_type.value,
                    })

        # Update milestone progress
        for milestone in project_milestones:
            await self.update_milestone_progress(milestone.id)

        return {
            "project": project.to_dict(),
            "progress_percentage": round(progress, 1),
            "task_stats": task_stats,
            "milestone_stats": milestone_stats,
            "tasks": [t.to_dict() for t in project_tasks],
            "milestones": [m.to_dict() for m in project_milestones],
            "linked_projects": linked,
        }

    # ==================== Cross-Project Features ====================

    async def link_projects(
        self,
        source_id: str,
        target_id: str,
        link_type: LinkType = LinkType.RELATED_TO,
        shared_concepts: Optional[list[str]] = None,
    ) -> Optional[ProjectLink]:
        """
        Create a link between two projects.

        Args:
            source_id: Source project ID
            target_id: Target project ID
            link_type: Type of relationship
            shared_concepts: Concepts shared between projects

        Returns:
            Created ProjectLink or None if projects not found
        """
        source = self.projects.get(source_id)
        target = self.projects.get(target_id)

        if not source or not target:
            logger.warning(f"One or both projects not found: {source_id}, {target_id}")
            return None

        # Check for existing link
        for link in self.links.values():
            if link.source_project_id == source_id and link.target_project_id == target_id:
                logger.info(f"Link already exists between {source_id} and {target_id}")
                return link

        link_id = str(uuid.uuid4())[:8]

        link = ProjectLink(
            id=link_id,
            source_project_id=source_id,
            target_project_id=target_id,
            link_type=link_type,
            shared_concepts=shared_concepts or [],
        )

        self.links[link_id] = link

        # Update project references
        if target_id not in source.linked_project_ids:
            source.linked_project_ids.append(target_id)
            self._save_project(source)

        self._save_link(link)

        logger.info(f"Linked projects: {source.name} -> {target.name} ({link_type.value})")
        return link

    async def find_similar_projects(
        self,
        project_id: str,
        limit: int = 5,
    ) -> list[tuple[Project, float]]:
        """
        Find projects similar to the given project based on embeddings.

        Args:
            project_id: Project to find similar projects for
            limit: Maximum number of results

        Returns:
            List of (Project, similarity_score) tuples
        """
        project = self.projects.get(project_id)
        if not project or not project.embedding:
            return []

        results = []

        for other_id, other in self.projects.items():
            if other_id == project_id or not other.embedding:
                continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(project.embedding, other.embedding)
            results.append((other, similarity))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def transfer_knowledge(
        self,
        from_id: str,
        to_id: str,
    ) -> Optional[dict]:
        """
        Transfer learnings from one project to another.

        Uses LLM to identify transferable knowledge and creates a link.

        Returns:
            Dict with transferred knowledge or None if failed
        """
        source = self.projects.get(from_id)
        target = self.projects.get(to_id)

        if not source or not target:
            logger.warning(f"One or both projects not found")
            return None

        if not self.llm_fn:
            logger.warning("LLM function not configured")
            return None

        try:
            # Get source project context
            source_tasks = [
                self.tasks[tid].to_dict() for tid in source.task_ids
                if tid in self.tasks
            ]

            prompt = f"""Analyze these two projects and identify knowledge that can be transferred from the source to the target.

SOURCE PROJECT: {source.name}
Description: {source.description}
Tags: {source.tags}
Tasks: {json.dumps(source_tasks, default=str)}

TARGET PROJECT: {target.name}
Description: {target.description}
Tags: {target.tags}

Identify:
1. Shared concepts between the projects
2. Learnings from the source that apply to the target
3. Patterns or approaches that could be reused

Respond in JSON format:
{{
    "shared_concepts": ["concept1", "concept2"],
    "transferable_learnings": ["learning1", "learning2"],
    "recommendations": ["recommendation1"]
}}"""

            response = await self.llm_fn(prompt)

            # Parse response
            if isinstance(response, str):
                import re
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    return None
            else:
                data = response

            # Create or update link
            link = await self.link_projects(
                source_id=from_id,
                target_id=to_id,
                link_type=LinkType.INFORMS,
                shared_concepts=data.get("shared_concepts", []),
            )

            if link:
                link.transferred_learnings = data.get("transferable_learnings", [])
                self._save_link(link)

            # Record transfer in projects
            transfer_record = {
                "from_project": from_id,
                "to_project": to_id,
                "timestamp": datetime.now().isoformat(),
                "learnings": data.get("transferable_learnings", []),
                "recommendations": data.get("recommendations", []),
            }

            target.knowledge_transfers.append(transfer_record)
            self._save_project(target)

            logger.info(f"Transferred knowledge from {source.name} to {target.name}")
            return data

        except Exception as e:
            logger.error(f"Knowledge transfer failed: {e}")
            return None

    # ==================== Statistics ====================

    def get_stats(self) -> dict:
        """Get project tracking statistics."""
        active_projects = [p for p in self.projects.values() if p.status == ProjectStatus.ACTIVE]

        return {
            "total_projects": len(self.projects),
            "active_projects": len(active_projects),
            "completed_projects": len([p for p in self.projects.values() if p.status == ProjectStatus.COMPLETED]),
            "total_tasks": len(self.tasks),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            "blocked_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.BLOCKED]),
            "total_milestones": len(self.milestones),
            "achieved_milestones": len([m for m in self.milestones.values() if m.is_achieved]),
            "project_links": len(self.links),
        }
