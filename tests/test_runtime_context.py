#!/usr/bin/env python3
"""
RuntimeContext tests for Hermes middleware.
Tests that RuntimeContext is properly constructed and immutable.
"""

import unittest
from hermes_cli.middleware import RuntimeContext


class TestRuntimeContext(unittest.TestCase):
    """Test RuntimeContext immutability and construction."""
    
    def test_runtime_context_immutable(self):
        """Verify that RuntimeContext objects are immutable."""
        ctx = RuntimeContext(
            task_id="test_task",
            session_id="test_session", 
            tool_call_id="test_tool_call",
            turn_id="test_turn",
            api_request_id="test_api_request",
            function_name="test_function",
            middleware_start_time=123.456
        )
        
        # Verify fields exist and are correct
        self.assertEqual(ctx.task_id, "test_task")
        self.assertEqual(ctx.session_id, "test_session")
        self.assertEqual(ctx.tool_call_id, "test_tool_call") 
        self.assertEqual(ctx.turn_id, "test_turn")
        self.assertEqual(ctx.api_request_id, "test_api_request")
        self.assertEqual(ctx.function_name, "test_function")
        self.assertEqual(ctx.middleware_start_time, 123.456)
        
        # Verify immutability - trying to modify should raise AttributeError
        with self.assertRaises(AttributeError):
            ctx.task_id = "modified"
            
    def test_runtime_context_creation(self):
        """Test that RuntimeContext can be created with required fields."""
        ctx = RuntimeContext(
            task_id="test_task",
            session_id="test_session", 
            tool_call_id="test_tool_call",
            turn_id="test_turn",
            api_request_id="test_api_request",
            function_name="test_function",
            middleware_start_time=123.456
        )
        
        # Verify all fields are present
        self.assertIsInstance(ctx.task_id, str)
        self.assertIsInstance(ctx.session_id, str)
        self.assertIsInstance(ctx.tool_call_id, str)
        self.assertIsInstance(ctx.turn_id, str) 
        self.assertIsInstance(ctx.api_request_id, str)
        self.assertIsInstance(ctx.function_name, str)
        self.assertIsInstance(ctx.middleware_start_time, float)
        self.assertEqual(ctx.middleware_start_time, 123.456)


if __name__ == '__main__':
    unittest.main()