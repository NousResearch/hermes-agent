"""Sending a message to a platform outside a running turn: ``hermes send`` and cron delivery.

``send`` routes one message through the live gateway adapter or a standalone sender. ``senders`` holds the
standalone per-platform senders and the error and result helpers. ``targets`` parses and resolves
``platform:chat_id[:thread]``.
"""
