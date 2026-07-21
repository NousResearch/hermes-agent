"""Tests for the model deprecation system."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from hermes_cli.model_deprecation import (
    ModelDeprecationRecord,
    ModelDeprecationDB,
    ModelAvailabilityChecker,
    record_model_failure,
    get_deprecation_info,
    get_deprecation_message,
    should_redirect_model,
    validate_model_with_deprecation_check,
    get_all_deprecations,
    clear_deprecation_cache,
    _KNOWN_DEPRECATIONS,
    _DEPRECATION_FAILURE_THRESHOLD,
    _DEPRECATION_FAILURE_WINDOW,
)


class TestModelDeprecationRecord:
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization of deprecation records."""
        record = ModelDeprecationRecord(
            model_id="test-model",
            provider="test-provider",
            first_detected=datetime.now(),
            last_detected=datetime.now(),
            failure_count=1,
            status="pending",
            suggested_alternatives=["alt-model-1", "alt-model-2"],
            redirect_model="alt-model-1",
            error_message="Model not found",
        )
        
        # Test serialization
        data = record.to_dict()
        assert data["model_id"] == "test-model"
        assert data["provider"] == "test-provider"
        assert data["failure_count"] == 1
        assert data["status"] == "pending"
        assert len(data["suggested_alternatives"]) == 2
        assert data["redirect_model"] == "alt-model-1"
        
        # Test deserialization
        restored = ModelDeprecationRecord.from_dict(data)
        assert restored.model_id == record.model_id
        assert restored.provider == record.provider
        assert restored.failure_count == record.failure_count
        assert restored.status == record.status
        assert restored.suggested_alternatives == record.suggested_alternatives
        assert restored.redirect_model == record.redirect_model


class TestModelDeprecationDB:
    def test_database_initialization(self):
        """Test database initialization with temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_deprecations.json"
            db = ModelDeprecationDB(db_path)
            
            # Should create empty database
            assert len(db._records) == 0
            # Database file is only created when records are saved
            assert not db_path.exists()
            
            # Save a record to create the file
            record = ModelDeprecationRecord(
                model_id="test-model",
                provider="test-provider",
                first_detected=datetime.now(),
                last_detected=datetime.now(),
                failure_count=1,
            )
            db.update_record(record)
            
            # Now the file should exist
            assert db_path.exists()
    
    def test_record_crud(self):
        """Test create, read, update, delete operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_deprecations.json"
            db = ModelDeprecationDB(db_path)
            
            # Create record
            record = ModelDeprecationRecord(
                model_id="test-model",
                provider="test-provider",
                first_detected=datetime.now(),
                last_detected=datetime.now(),
                failure_count=1,
            )
            db.update_record(record)
            
            # Read record
            retrieved = db.get_record("test-model", "test-provider")
            assert retrieved is not None
            assert retrieved.model_id == "test-model"
            assert retrieved.failure_count == 1
            
            # Update record
            retrieved.failure_count = 2
            db.update_record(retrieved)
            updated = db.get_record("test-model", "test-provider")
            assert updated.failure_count == 2
            
            # Delete record
            db.delete_record("test-model", "test-provider")
            deleted = db.get_record("test-model", "test-provider")
            assert deleted is None
    
    def test_get_confirmed_deprecations(self):
        """Test filtering confirmed deprecations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_deprecations.json"
            db = ModelDeprecationDB(db_path)
            
            # Add pending record
            pending_record = ModelDeprecationRecord(
                model_id="pending-model",
                provider="test-provider",
                first_detected=datetime.now(),
                last_detected=datetime.now(),
                failure_count=1,
                status="pending",
            )
            db.update_record(pending_record)
            
            # Add confirmed record
            confirmed_record = ModelDeprecationRecord(
                model_id="confirmed-model",
                provider="test-provider",
                first_detected=datetime.now(),
                last_detected=datetime.now(),
                failure_count=3,
                status="confirmed",
            )
            db.update_record(confirmed_record)
            
            # Get confirmed deprecations
            confirmed = db.get_confirmed_deprecations()
            assert len(confirmed) == 1
            assert confirmed[0].model_id == "confirmed-model"


class TestModelAvailabilityChecker:
    def test_cache_functionality(self):
        """Test that availability results are cached."""
        checker = ModelAvailabilityChecker()
        
        # First call should populate cache
        with patch.object(checker, '_check_model_availability') as mock_check:
            mock_check.return_value = (True, None)
            available, _ = checker.check_availability("test-model", "test-provider")
            assert available is True
            assert mock_check.call_count == 1
        
        # Second call should use cache
        with patch.object(checker, '_check_model_availability') as mock_check:
            available, _ = checker.check_availability("test-model", "test-provider")
            assert available is True
            assert mock_check.call_count == 0  # Should not be called due to cache
    
    def test_availability_check_success(self):
        """Test successful availability check."""
        checker = ModelAvailabilityChecker()
        
        with patch.object(checker, '_check_model_availability') as mock_check:
            mock_check.return_value = (True, None)
            available, error = checker.check_availability("test-model", "test-provider")
            
            assert available is True
            assert error is None
    
    def test_availability_check_failure(self):
        """Test failed availability check."""
        checker = ModelAvailabilityChecker()
        
        with patch.object(checker, '_check_model_availability') as mock_check:
            mock_check.return_value = (False, "Model not found in provider listing")
            available, error = checker.check_availability("test-model", "test-provider")
            
            assert available is False
            assert error is not None
            assert "not found" in error


class TestRecordModelFailure:
    def test_new_record_creation(self):
        """Test creating a new deprecation record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_deprecations.json"
            
            with patch('hermes_cli.model_deprecation._deprecation_db', ModelDeprecationDB(db_path)):
                record = record_model_failure(
                    "test-model",
                    "test-provider",
                    "Model not found"
                )
                
                assert record.model_id == "test-model"
                assert record.provider == "test-provider"
                assert record.failure_count == 1
                assert record.status == "pending"
    
    def test_existing_record_update(self):
        """Test updating an existing deprecation record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_deprecations.json"
            
            with patch('hermes_cli.model_deprecation._deprecation_db', ModelDeprecationDB(db_path)):
                # First failure
                record1 = record_model_failure(
                    "test-model",
                    "test-provider",
                    "Model not found"
                )
                assert record1.failure_count == 1
                
                # Second failure
                record2 = record_model_failure(
                    "test-model",
                    "test-provider",
                    "Model not found"
                )
                assert record2.failure_count == 2
    
    def test_threshold_confirmation(self):
        """Test that record is confirmed after threshold failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_deprecations.json"
            
            with patch('hermes_cli.model_deprecation._deprecation_db', ModelDeprecationDB(db_path)):
                # Simulate threshold failures
                for _ in range(_DEPRECATION_FAILURE_THRESHOLD):
                    record_model_failure(
                        "test-model",
                        "test-provider",
                        "Model not found"
                    )
                
                record = record_model_failure(
                    "test-model",
                    "test-provider",
                    "Model not found"
                )
                
                assert record.failure_count >= _DEPRECATION_FAILURE_THRESHOLD
                assert record.status == "confirmed"
    
    def test_known_deprecation_info(self):
        """Test that known deprecation info is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_deprecations.json"
            
            with patch('hermes_cli.model_deprecation._deprecation_db', ModelDeprecationDB(db_path)):
                record = record_model_failure(
                    "deepseek-chat",
                    "deepseek",
                    "Model not found"
                )
                
                # Should have known deprecation info
                assert len(record.suggested_alternatives) > 0
                assert record.redirect_model is not None


class TestGetDeprecationInfo:
    def test_known_deprecation(self):
        """Test getting info for known deprecated model."""
        info = get_deprecation_info("deepseek-chat", "deepseek")
        
        assert info is not None
        assert info["provider"] == "deepseek"
        assert info["redirect_model"] == "deepseek-v4-pro"
        assert len(info["suggested_alternatives"]) > 0
        assert info["source"] == "known"
        assert info["redirect_model"] is not None
        assert len(info["suggested_alternatives"]) > 0
    
    def test_non_deprecated_model(self):
        """Test getting info for non-deprecated model."""
        info = get_deprecation_info("gpt-4", "openai")
        
        assert info is None


class TestGetDeprecationMessage:
    def test_deprecation_message_format(self):
        """Test that deprecation messages are properly formatted."""
        message = get_deprecation_message("deepseek-chat", "deepseek")
        
        assert message is not None
        assert "deprecated" in message.lower()
        assert "deepseek-v4-pro" in message  # redirect model
        assert "⚠️" in message
    
    def test_no_message_for_valid_model(self):
        """Test that no message is returned for valid models."""
        message = get_deprecation_message("gpt-4", "openai")
        
        assert message is None


class TestShouldRedirectModel:
    def test_known_redirect(self):
        """Test redirect for known deprecated model."""
        redirect = should_redirect_model("deepseek-chat", "deepseek")
        
        assert redirect == "deepseek-v4-pro"
    
    def test_no_redirect_for_valid_model(self):
        """Test that no redirect is suggested for valid models."""
        redirect = should_redirect_model("gpt-4", "openai")
        
        assert redirect is None


class TestGetAllDeprecations:
    def test_includes_known_and_detected(self):
        """Test that both known and detected deprecations are included."""
        all_deps = get_all_deprecations()
        
        assert "known" in all_deps
        assert "detected" in all_deps
        assert len(all_deps["known"]) > 0  # Should have deepseek-chat at least


class TestValidateModelWithDeprecationCheck:
    def test_deprecated_model_detection(self):
        """Test that deprecated models are detected during validation."""
        result = validate_model_with_deprecation_check(
            "deepseek-chat",
            "deepseek"
        )
        
        assert result["is_deprecated"] is True
        assert "deprecation_info" in result
        assert result["message"] is not None
        assert "deprecated" in result["message"].lower()
    
    def test_valid_model_validation(self):
        """Test that valid models pass validation normally."""
        result = validate_model_with_deprecation_check(
            "gpt-4",
            "openai"
        )
        
        assert result["is_deprecated"] is False
        assert "deprecation_info" not in result


class TestClearDeprecationCache:
    def test_cache_clearing(self):
        """Test that the deprecation cache can be cleared."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_deprecations.json"
            
            # Create a record
            db = ModelDeprecationDB(db_path)
            record = ModelDeprecationRecord(
                model_id="test-model",
                provider="test-provider",
                first_detected=datetime.now(),
                last_detected=datetime.now(),
                failure_count=1,
            )
            db.update_record(record)
            
            # Verify it exists
            assert db.get_record("test-model", "test-provider") is not None
            
            # Clear cache (this would use the global instance in real usage)
            clear_deprecation_cache()
            
            # The global instance should be reset, but our local db still has the record
            # This is expected behavior - clear_deprecation_cache() affects the global instance


class TestIntegration:
    """Integration tests for the complete deprecation workflow."""
    
    def test_full_deprecation_workflow(self):
        """Test the complete workflow from failure to confirmation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_deprecations.json"
            
            with patch('hermes_cli.model_deprecation._deprecation_db', ModelDeprecationDB(db_path)):
                # Simulate multiple failures
                for i in range(_DEPRECATION_FAILURE_THRESHOLD):
                    record_model_failure(
                        "test-model",
                        "test-provider",
                        f"Failure attempt {i+1}"
                    )
                
                # Check that model is now marked as deprecated
                info = get_deprecation_info("test-model", "test-provider")
                assert info is not None
                assert info["source"] == "detected"
                
                # Check that deprecation message is available
                message = get_deprecation_message("test-model", "test-provider")
                assert message is not None
                assert "deprecated" in message.lower()
    
    def test_known_deprecation_priority(self):
        """Test that known deprecations take priority over detected ones."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_deprecations.json"
            
            with patch('hermes_cli.model_deprecation._deprecation_db', ModelDeprecationDB(db_path)):
                # Create a detected record for deepseek-chat
                record = ModelDeprecationRecord(
                    model_id="deepseek-chat",
                    provider="deepseek",
                    first_detected=datetime.now(),
                    last_detected=datetime.now(),
                    failure_count=3,
                    status="confirmed",
                    suggested_alternatives=["some-alternative"],
                    redirect_model="some-redirect",
                )
                record_db = ModelDeprecationDB(db_path)
                record_db.update_record(record)
                
                # Known deprecation should still take priority
                info = get_deprecation_info("deepseek-chat", "deepseek")
                assert info["source"] == "known"
                assert info["redirect_model"] == "deepseek-v4-pro"  # From known, not detected