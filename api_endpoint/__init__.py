"""
Hermes Agent - API Endpoint & Real-time Logging

This package provides a FastAPI WebSocket endpoint for real-time logging of the Hermes Agent.

Components:
- logging_server: FastAPI server that receives and stores events
- websocket_logger: Client library for sending events from the agent

Usage:
    # Start the API endpoint server
    python api_endpoint/logging_server.py
    
    # Use in agent code
    from api_endpoint.websocket_logger import WebSocketLogger
    
For more information, see:
- WEBSOCKET_LOGGING_GUIDE.md - User guide
- IMPLEMENTATION_SUMMARY.md - Technical details
"""

from .websocket_logger import WebSocketLogger, SyncWebSocketLogger

__all__ = ['WebSocketLogger', 'SyncWebSocketLogger']
__version__ = '1.0.0'

