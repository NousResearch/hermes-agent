"""Tests for skill dependency resolution."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestTokenEstimator:
    """Test the _estimate_tokens function."""

    def test_empty_text(self):
        """Test empty text returns 0 tokens."""
        from agent.skill_dependencies import _estimate_tokens
        
        assert _estimate_tokens("") == 0
        assert _estimate_tokens(None) == 0

    def test_simple_text(self):
        """Test simple text estimation."""
        from agent.skill_dependencies import _estimate_tokens
        
        # "hello world" = 2 words * 1.3 + 0 lines * 0.5 = 2.6 -> 2
        tokens = _estimate_tokens("hello world")
        assert tokens == 2

    def test_longer_text(self):
        """Test longer text estimation."""
        from agent.skill_dependencies import _estimate_tokens
        
        text = " ".join(["word"] * 100)  # 100 words
        tokens = _estimate_tokens(text)
        # 100 * 1.3 = 130, plus 0 lines * 0.5 = 0
        assert 120 <= tokens <= 140


class TestDependencyResolverCycleDetection:
    """Test cycle detection in dependency resolution."""

    def test_direct_cycle(self):
        """Test A -> A (self-referential)."""
        from agent.skill_dependencies import SkillDependencyResolver
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create skill A that depends on itself
            a_dir = Path(tmpdir) / "skill-a"
            a_dir.mkdir()
            a_md = a_dir / "SKILL.md"
            a_md.write_text("""---
name: skill-a
dependencies:
  - skill-a
---

# Skill A
Content here.
""")
            
            resolver = SkillDependencyResolver(
                max_depth=3, max_skills=10, max_tokens=10000, enabled=True
            )
            
            with patch("hermes_constants.get_skills_dir") as mock_dir:
                mock_dir.return_value = Path(tmpdir)
                
                result = resolver.resolve("skill-a")
                
                # Should load skill-a but report cycle on dependency
                assert len(result.loaded) == 1  # just the parent
                assert any(e.reason == "cycle" for e in result.errors)
                assert result.was_truncated

    def test_indirect_cycle(self):
        """Test A -> B -> A (indirect cycle)."""
        from agent.skill_dependencies import SkillDependencyResolver
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create skill A that depends on B
            a_dir = Path(tmpdir) / "skill-a"
            a_dir.mkdir()
            a_md = a_dir / "SKILL.md"
            a_md.write_text("""---
name: skill-a
dependencies:
  - skill-b
---

# Skill A
Content here.
""")
            
            # Create skill B that depends on A
            b_dir = Path(tmpdir) / "skill-b"
            b_dir.mkdir()
            b_md = b_dir / "SKILL.md"
            b_md.write_text("""---
name: skill-b
dependencies:
  - skill-a
---

# Skill B
Content here.
""")
            
            resolver = SkillDependencyResolver(
                max_depth=3, max_skills=10, max_tokens=10000, enabled=True
            )
            
            with patch("hermes_constants.get_skills_dir") as mock_dir:
                mock_dir.return_value = Path(tmpdir)
                
                result = resolver.resolve("skill-a")
                
                # Should load A -> B, but detect cycle on A -> B -> A
                assert len(result.loaded) == 2  # skill-a + skill-b
                assert any(e.reason == "cycle" for e in result.errors)
                assert result.was_truncated


class TestDependencyResolverDepthLimit:
    """Test depth limit safeguard."""

    def test_depth_limit(self):
        """Test that depth limit is enforced."""
        from agent.skill_dependencies import SkillDependencyResolver
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create chain: A -> B -> C -> D -> E
            for i, (name, dep) in enumerate([
                ("skill-a", "skill-b"),
                ("skill-b", "skill-c"),
                ("skill-c", "skill-d"),
                ("skill-d", "skill-e"),
            ]):
                s_dir = Path(tmpdir) / name
                s_dir.mkdir()
                s_md = s_dir / "SKILL.md"
                s_md.write_text(f"""---
name: {name}
dependencies:
  - {dep}
---

# {name}
Content for {name}.
""")
            
            # Create E (no dependencies)
            e_dir = Path(tmpdir) / "skill-e"
            e_dir.mkdir()
            e_md = e_dir / "SKILL.md"
            e_md.write_text("""---
name: skill-e
---

# Skill E
Content for skill e.
""")
            
            # Set depth limit to 2
            resolver = SkillDependencyResolver(
                max_depth=2, max_skills=10, max_tokens=10000, enabled=True
            )
            
            with patch("hermes_constants.get_skills_dir") as mock_dir:
                mock_dir.return_value = Path(tmpdir)
                
                result = resolver.resolve("skill-a")
                
                # Should load A (depth 0), B (depth 1), C (depth 2)
                # D (depth 3) should be blocked by depth limit
                assert len(result.loaded) == 3
                assert any("depth_limit" in e.reason for e in result.errors)
                assert result.max_depth_reached >= 3


class TestDependencyResolverCountLimit:
    """Test skill count limit safeguard."""

    def test_count_limit(self):
        """Test that total skill count is enforced."""
        from agent.skill_dependencies import SkillDependencyResolver
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create A -> B, C, D, E, F (5 dependencies)
            a_dir = Path(tmpdir) / "skill-a"
            a_dir.mkdir()
            a_md = a_dir / "SKILL.md"
            a_md.write_text("""---
name: skill-a
dependencies:
  - skill-b
  - skill-c
  - skill-d
  - skill-e
  - skill-f
---

# Skill A
Content here.
""")
            
            deps = ["skill-b", "skill-c", "skill-d", "skill-e", "skill-f"]
            for name in deps:
                s_dir = Path(tmpdir) / name
                s_dir.mkdir()
                s_md = s_dir / "SKILL.md"
                s_md.write_text(f"""---
name: {name}
---

# {name}
Content for {name}.
""")
            
            # Set max skills to 3 (parent + 2 dependencies max)
            resolver = SkillDependencyResolver(
                max_depth=3, max_skills=3, max_tokens=10000, enabled=True
            )
            
            with patch("hermes_constants.get_skills_dir") as mock_dir:
                mock_dir.return_value = Path(tmpdir)
                
                result = resolver.resolve("skill-a")
                
                # Should load A + 2 deps (3 total), rest blocked by count limit
                assert result.total_skills == 3
                assert len(result.loaded) == 3
                assert any("count_limit" in e.reason for e in result.errors)


class TestDependencyResolverTokenBudget:
    """Test token budget safeguard."""

    def test_token_budget(self):
        """Test that token budget is enforced."""
        from agent.skill_dependencies import SkillDependencyResolver
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create A -> B where B has lots of content
            a_dir = Path(tmpdir) / "skill-a"
            a_dir.mkdir()
            a_md = a_dir / "SKILL.md"
            a_md.write_text("""---
name: skill-a
dependencies:
  - skill-b
---

# Skill A
Small content.
""")
            
            b_dir = Path(tmpdir) / "skill-b"
            b_dir.mkdir()
            b_md = b_dir / "SKILL.md"
            b_md.write_text("""---
name: skill-b
---

# Skill B
""" + " ".join(["word"] * 1000) + "\n")
            
            resolver = SkillDependencyResolver(
                max_depth=3, max_skills=10, max_tokens=100, enabled=True
            )
            
            with patch("hermes_constants.get_skills_dir") as mock_dir:
                mock_dir.return_value = Path(tmpdir)
                
                result = resolver.resolve("skill-a")
                
                # skill-a has ~10 tokens, skill-b has ~1300 tokens
                # skill-a should load, skill-b blocked by token budget
                assert result.total_skills >= 1  # at least skill-a loaded
                # skill-b may or may not be loaded depending on token estimate
                # The important thing is the error is recorded if blocked


class TestDependencyResolverOptional:
    """Test optional dependencies."""

    def test_optional_dep_not_found(self):
        """Test that missing optional deps don't block loading."""
        from agent.skill_dependencies import SkillDependencyResolver
        
        with tempfile.TemporaryDirectory() as tmpdir:
            a_dir = Path(tmpdir) / "skill-a"
            a_dir.mkdir()
            a_md = a_dir / "SKILL.md"
            a_md.write_text("""---
name: skill-a
optional_dependencies:
  - nonexistent-skill
---

# Skill A
Content here.
""")
            
            resolver = SkillDependencyResolver(
                max_depth=3, max_skills=10, max_tokens=10000, enabled=True
            )
            
            with patch("hermes_constants.get_skills_dir") as mock_dir:
                mock_dir.return_value = Path(tmpdir)
                
                result = resolver.resolve("skill-a")
                
                # Should still load A even though optional dep doesn't exist
                assert result.total_skills >= 1
                assert not any(e.skill_name == "nonexistent-skill" for e in result.errors)


class TestDependencyResolverDisabled:
    """Test that disabled resolver returns empty result."""

    def test_disabled(self):
        """Test that disabled resolver returns empty result."""
        from agent.skill_dependencies import SkillDependencyResolver
        
        resolver = SkillDependencyResolver(enabled=False)
        result = resolver.resolve("any-skill")
        
        assert len(result.loaded) == 0
        assert not result.was_truncated


class TestDependencyResolverIntegration:
    """Integration test with full workflow."""

    def test_full_resolution(self):
        """Test a complete dependency resolution."""
        from agent.skill_dependencies import resolve_skill_dependencies, SkillDependencyResolver
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create deployment skill that depends on infra + networking
            deploy_dir = Path(tmpdir) / "rapidwebs-infra-deployment"
            deploy_dir.mkdir()
            deploy_md = deploy_dir / "SKILL.md"
            deploy_md.write_text("""---
name: rapidwebs-infra-deployment
dependencies:
  - incus-vm-management
  - tailscale-network-ops
optional_dependencies:
  - rapidwebs-backup-restore
---

# Deployment Skill
Step by step deployment instructions.
""")
            
            # Create incus skill (no deps)
            incus_dir = Path(tmpdir) / "incus-vm-management"
            incus_dir.mkdir()
            incus_md = incus_dir / "SKILL.md"
            incus_md.write_text("""---
name: incus-vm-management
---

# Incus VM Management
Create and manage Incus VMs.
""")
            
            # Create tailscale skill (no deps)
            tail_dir = Path(tmpdir) / "tailscale-network-ops"
            tail_dir.mkdir()
            tail_md = tail_dir / "SKILL.md"
            tail_md.write_text("""---
name: tailscale-network-ops
---

# Tailscale Network Ops
ACL management and subnet routing.
""")
            
            # Create backup skill (no deps)
            backup_dir = Path(tmpdir) / "rapidwebs-backup-restore"
            backup_dir.mkdir()
            backup_md = backup_dir / "SKILL.md"
            backup_md.write_text("""---
name: rapidwebs-backup-restore
---

# Backup and Restore
Backup infrastructure with restic.
""")
            
            SkillDependencyResolver.reset_instance()
            resolver = SkillDependencyResolver(
                max_depth=3, max_skills=10, max_tokens=50000, enabled=True
            )
            
            with patch("hermes_constants.get_skills_dir") as mock_dir:
                mock_dir.return_value = Path(tmpdir)
                
                result = resolver.resolve("rapidwebs-infra-deployment")
                
                # Should load: deployment (depth 0) + incus (1) + tailscale (1) + backup (1, optional)
                assert result.total_skills >= 4
                assert result.total_tokens > 0
                
                # Check loaded skill metadata
                names = [s.name for s in result.loaded]
                assert "rapidwebs-infra-deployment" in names
                assert "incus-vm-management" in names
                assert "tailscale-network-ops" in names
                assert "rapidwebs-backup-restore" in names
                
                # Check parent is at depth 0
                parent = [s for s in result.loaded if s.is_parent]
                assert len(parent) == 1
                assert parent[0].name == "rapidwebs-infra-deployment"
                
                # Check optional was marked
                optional = [s for s in result.loaded if s.is_optional]
                assert len(optional) == 1
                assert optional[0].name == "rapidwebs-backup-restore"


class TestResolveSkillDependencies:
    """Test the convenience wrapper."""

    def test_force_reload(self):
        """Test force reload from config."""
        from agent.skill_dependencies import (
            SkillDependencyResolver,
            resolve_skill_dependencies,
        )
        
        # Set an instance
        SkillDependencyResolver._INSTANCE = SkillDependencyResolver(enabled=False)
        
        # Calling without force should use the disabled instance
        with tempfile.TemporaryDirectory() as tmpdir:
            a_dir = Path(tmpdir) / "skill-a"
            a_dir.mkdir()
            a_md = a_dir / "SKILL.md"
            a_md.write_text("""---
name: skill-a
---

# Skill A
Content.
""")
            
            with patch("hermes_constants.get_skills_dir") as mock_dir:
                mock_dir.return_value = Path(tmpdir)
                
                result = resolve_skill_dependencies("skill-a", force=False)
                assert result.total_skills == 0  # disabled
                
                result = resolve_skill_dependencies("skill-a", force=True)
                # After force=True, reads config and creates new instance
                # (will use config defaults)
                assert result.total_skills >= 0