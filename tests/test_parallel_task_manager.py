"""
Tests for the Parallel Task Manager and Task Classifier.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from gateway.task_classifier import (
    TaskClassifier,
    TaskType,
    classify_task,
)
from gateway.parallel_task_manager import (
    ParallelTaskManager,
    ParallelTask,
    TaskStatus,
)


class TestTaskClassifier:
    """Tests for task classification."""
    
    def test_classify_independent_search(self):
        """Test classification of search tasks."""
        classifier = TaskClassifier()
        result = classifier.classify("Search for Python async patterns")
        
        assert result.task_type == TaskType.INDEPENDENT
        assert result.confidence > 0.6
        assert "search" in result.reasoning.lower()
    
    def test_classify_independent_image(self):
        """Test classification of image generation."""
        classifier = TaskClassifier()
        result = classifier.classify("Generate an image of a cat")
        
        assert result.task_type == TaskType.INDEPENDENT
        assert result.confidence > 0.6
    
    def test_classify_sequential_code(self):
        """Test classification of code editing."""
        classifier = TaskClassifier()
        result = classifier.classify("Edit the main.py file to add logging")
        
        assert result.task_type == TaskType.SEQUENTIAL
        assert result.confidence > 0.5
    
    def test_classify_dependent(self):
        """Test classification of dependent tasks."""
        classifier = TaskClassifier()
        result = classifier.classify(
            "Based on those results, update the configuration",
            running_tasks=["previous search task"]
        )
        
        assert result.task_type == TaskType.DEPENDENT
        assert result.confidence > 0.5
    
    def test_classify_blocking(self):
        """Test classification of blocking tasks."""
        classifier = TaskClassifier()
        result = classifier.classify("Should I proceed with the deletion?")
        
        assert result.task_type == TaskType.BLOCKING
    
    def test_parallel_intent_detection(self):
        """Test detection of explicit parallel intent."""
        classifier = TaskClassifier()
        result = classifier.classify(
            "In parallel, can you search for documentation while I think about the design?"
        )
        
        assert result.task_type == TaskType.INDEPENDENT
        assert result.confidence > 0.7
    
    def test_can_run_in_parallel_independent(self):
        """Test parallel check for independent tasks."""
        classifier = TaskClassifier()
        
        task1 = classifier.classify("Search for Python docs")
        task2 = classifier.classify("Search for JavaScript docs")
        
        can_parallel = classifier.can_run_in_parallel(
            MagicMock(classification=task1, suggested_toolsets={"web"}),
            [MagicMock(classification=task2, suggested_toolsets={"web"})]
        )
        
        assert can_parallel is True
    
    def test_cannot_run_parallel_sequential(self):
        """Test that sequential tasks don't run in parallel."""
        classifier = TaskClassifier()
        
        new_task = classifier.classify("Edit the main.py file")
        running_task = classifier.classify("Modify the utils.py file")
        
        can_parallel = classifier.can_run_in_parallel(
            MagicMock(classification=new_task, suggested_toolsets={"file"}),
            [MagicMock(classification=running_task, suggested_toolsets={"file"})]
        )
        
        assert can_parallel is False


class TestParallelTaskManager:
    """Tests for the parallel task manager."""
    
    @pytest.fixture
    async def manager(self):
        """Create a task manager for testing."""
        mgr = ParallelTaskManager(max_concurrent=2)
        await mgr.start()
        yield mgr
        await mgr.stop()
    
    @pytest.mark.asyncio
    async def test_submit_and_complete_task(self, manager):
        """Test submitting and completing a task."""
        async def mock_runner(task):
            await asyncio.sleep(0.1)
            return {"result": "success"}
        
        task = await manager.submit_task(
            session_key="test_session",
            message="Test task",
            task_runner=mock_runner
        )
        
        assert task.status == TaskStatus.PENDING
        
        # Wait for completion
        await asyncio.sleep(0.2)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_multiple_parallel_tasks(self, manager):
        """Test running multiple tasks in parallel."""
        execution_order = []
        
        async def mock_runner(task):
            execution_order.append((task.task_id, "start"))
            await asyncio.sleep(0.1)
            execution_order.append((task.task_id, "end"))
            return {"task_id": task.task_id}
        
        # Submit 3 tasks
        tasks = []
        for i in range(3):
            task = await manager.submit_task(
                session_key="test_session",
                message=f"Task {i}",
                task_runner=mock_runner
            )
            tasks.append(task)
        
        # Wait for all to complete
        await asyncio.sleep(0.5)
        
        # Check that tasks overlapped (parallel execution)
        starts = [e for e in execution_order if e[1] == "start"]
        ends = [e for e in execution_order if e[1] == "end"]
        
        assert len(starts) == 3
        assert len(ends) == 3
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, manager):
        """Test cancelling a running task."""
        async def slow_runner(task):
            await asyncio.sleep(10)  # Long running
            return {"result": "should not complete"}
        
        task = await manager.submit_task(
            session_key="test_session",
            message="Slow task",
            task_runner=slow_runner
        )
        
        # Wait a bit for task to start
        await asyncio.sleep(0.1)
        
        # Cancel it
        cancelled = await manager.cancel_task(task.task_id)
        assert cancelled is True
        
        # Wait and check
        await asyncio.sleep(0.2)
        assert task.status == TaskStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_get_session_tasks(self, manager):
        """Test getting tasks for a session."""
        async def mock_runner(task):
            return {"result": "ok"}
        
        # Submit tasks to different sessions
        task1 = await manager.submit_task(
            session_key="session_a",
            message="Task A1",
            task_runner=mock_runner
        )
        task2 = await manager.submit_task(
            session_key="session_a",
            message="Task A2",
            task_runner=mock_runner
        )
        task3 = await manager.submit_task(
            session_key="session_b",
            message="Task B1",
            task_runner=mock_runner
        )
        
        # Get session A tasks
        session_a_tasks = manager.get_session_tasks("session_a")
        assert len(session_a_tasks) == 2
        
        # Get session B tasks
        session_b_tasks = manager.get_session_tasks("session_b")
        assert len(session_b_tasks) == 1
    
    @pytest.mark.asyncio
    async def test_task_progress_updates(self, manager):
        """Test task progress tracking."""
        async def progress_runner(task):
            task.progress = 0.5
            task.progress_message = "Halfway done"
            await asyncio.sleep(0.1)
            task.progress = 1.0
            return {"result": "complete"}
        
        task = await manager.submit_task(
            session_key="test_session",
            message="Progress task",
            task_runner=progress_runner
        )
        
        await asyncio.sleep(0.2)
        
        assert task.progress == 1.0
        assert task.status == TaskStatus.COMPLETED
    
    def test_stats(self, manager):
        """Test getting manager statistics."""
        stats = manager.get_stats()
        
        assert "total_tasks" in stats
        assert "running_tasks" in stats
        assert "max_concurrent" in stats
        assert stats["max_concurrent"] == 2


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_parallel_execution(self):
        """Test complete parallel execution flow."""
        manager = ParallelTaskManager(max_concurrent=3)
        await manager.start()
        
        try:
            results = []
            
            async def task_runner(task):
                # Simulate work
                await asyncio.sleep(0.1)
                results.append(task.message)
                return {"completed": True}
            
            # Submit independent tasks
            tasks = []
            for i in range(3):
                task = await manager.submit_task(
                    session_key="test",
                    message=f"Task {i}",
                    task_runner=task_runner
                )
                tasks.append(task)
            
            # Wait for all
            await asyncio.sleep(0.5)
            
            assert len(results) == 3
            assert all(t.status == TaskStatus.COMPLETED for t in tasks)
            
        finally:
            await manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
