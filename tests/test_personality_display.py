"""Tests for /personality command display with current personality indicator."""

import pytest
from cli import HermesCLI


def test_get_current_personality_none():
    """Test that empty system_prompt returns 'none'."""
    cli = HermesCLI()
    cli.system_prompt = ""
    cli.personalities = {
        "kawaii": "You are a kawaii assistant!",
        "professional": "You are a professional assistant.",
    }
    current = cli._get_current_personality()
    assert current == "none"


def test_get_current_personality_none_value():
    """Test that None system_prompt returns 'none'."""
    cli = HermesCLI()
    cli.system_prompt = None
    cli.personalities = {
        "kawaii": "You are a kawaii assistant!",
    }
    current = cli._get_current_personality()
    assert current == "none"


def test_get_current_personality_kawaii():
    """Test that kawaii system_prompt returns 'kawaii'."""
    cli = HermesCLI()
    cli.system_prompt = "You are a kawaii assistant!"
    cli.personalities = {
        "kawaii": "You are a kawaii assistant!",
        "professional": "You are a professional assistant.",
    }
    current = cli._get_current_personality()
    assert current == "kawaii"


def test_get_current_personality_professional():
    """Test that professional system_prompt returns 'professional'."""
    cli = HermesCLI()
    cli.system_prompt = "You are a professional assistant."
    cli.personalities = {
        "kawaii": "You are a kawaii assistant!",
        "professional": "You are a professional assistant.",
    }
    current = cli._get_current_personality()
    assert current == "professional"


def test_get_current_personality_technical():
    """Test that technical system_prompt returns 'technical'."""
    cli = HermesCLI()
    cli.system_prompt = "You are a technical expert."
    cli.personalities = {
        "technical": "You are a technical expert.",
    }
    current = cli._get_current_personality()
    assert current == "technical"


def test_get_current_personality_custom():
    """Test that unmatched system_prompt returns 'custom'."""
    cli = HermesCLI()
    cli.personalities = {
        "kawaii": "You are a kawaii assistant!",
    }
    cli.system_prompt = "You are a custom assistant."
    current = cli._get_current_personality()
    assert current == "custom"


def test_get_current_personality_with_tone():
    """Test that system_prompt with tone prefix is matched correctly."""
    cli = HermesCLI()
    cli.personalities = {
        "kawaii": {
            "description": "A kawaii personality",
            "system_prompt": "You are a kawaii assistant!",
            "tone": "enthusiastic",
        }
    }
    cli.system_prompt = "You are a kawaii assistant!\nTone: enthusiastic"
    current = cli._get_current_personality()
    assert current == "kawaii"


def test_get_current_personality_with_style():
    """Test that system_prompt with style prefix is matched correctly."""
    cli = HermesCLI()
    cli.personalities = {
        "kawaii": {
            "description": "A kawaii personality",
            "system_prompt": "You are a kawaii assistant!",
            "style": "cute",
        }
    }
    cli.system_prompt = "You are a kawaii assistant!\nStyle: cute"
    current = cli._get_current_personality()
    assert current == "kawaii"


def test_get_current_personality_with_tone_and_style():
    """Test that system_prompt with both tone and style is matched correctly."""
    cli = HermesCLI()
    cli.personalities = {
        "kawaii": {
            "description": "A kawaii personality",
            "system_prompt": "You are a kawaii assistant!",
            "tone": "enthusiastic",
            "style": "cute",
        }
    }
    cli.system_prompt = "You are a kawaii assistant!\nTone: enthusiastic\nStyle: cute"
    current = cli._get_current_personality()
    assert current == "kawaii"


def test_get_current_personality_multiple_personalities():
    """Test that first matching personality is returned."""
    cli = HermesCLI()
    cli.system_prompt = "You are a professional assistant."
    cli.personalities = {
        "kawaii": "You are a kawaii assistant!",
        "professional": "You are a professional assistant.",
    }
    current = cli._get_current_personality()
    assert current == "professional"


def test_personality_display_current_line():
    """Test the current personality display line formatting."""
    cli = HermesCLI()
    cli.system_prompt = "You are a kawaii assistant!"
    cli.personalities = {
        "kawaii": "You are a kawaii assistant!",
        "professional": "You are a professional assistant.",
    }
    
    current = cli._get_current_personality()
    formatted = f"  Current:   {current:^12}"
    
    # Verify the formatting contains the current personality
    assert "Current:" in formatted
    assert "kawaii" in formatted


def test_personality_display_none():
    """Test that display shows 'none' when no personality is set."""
    cli = HermesCLI()
    cli.system_prompt = ""
    cli.personalities = {
        "kawaii": "You are a kawaii assistant!",
    }
    
    current = cli._get_current_personality()
    formatted = f"  Current:   {current:^12}"
    
    # Verify the formatting contains 'none'
    assert "Current:" in formatted
    assert "none" in formatted


def test_personality_display_custom():
    """Test that display shows 'custom' for custom prompts."""
    cli = HermesCLI()
    cli.system_prompt = "You are a custom assistant."
    cli.personalities = {
        "kawaii": "You are a kawaii assistant!",
    }
    
    current = cli._get_current_personality()
    formatted = f"  Current:   {current:^12}"
    
    # Verify the formatting contains 'custom'
    assert "Current:" in formatted
    assert "custom" in formatted
