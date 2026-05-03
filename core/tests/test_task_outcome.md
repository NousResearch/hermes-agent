# Test Documentation: TaskOutcome and TaskOutcomeStore

## Overview

This document provides test cases for the `TaskOutcome` dataclass and `TaskOutcomeStore` class. These components are responsible for recording, retrieving, and analyzing task execution outcomes with taxonomy-based categorization and statistics aggregation.

---

## Test Case Table

| ID | Description | Input | Expected Output |
|----|-------------|-------|-----------------|
| TO-001 | Record a successful task outcome | TaskOutcome(success=True, task_type="code_generation", taxonomy={"category": "python", "subcategory": "function"}) | Outcome stored with timestamp, returned with matching ID |
| TO-002 | Record a failed task outcome | TaskOutcome(success=False, task_type="testing", error_message="AssertionError: expected 200, got 404") | Outcome stored with error details and timestamp |
| TO-003 | Record outcome with context data | TaskOutcome(success=True, context={"lines_of_code": 150, "duration_ms": 2500}) | Context data preserved in stored outcome |
| TO-004 | Retrieve recent outcomes | get_recent(n=5) | List of 5 most recent TaskOutcome objects, sorted by timestamp descending |
| TO-005 | Retrieve recent outcomes with limit | get_recent(n=10) | List of up to 10 recent outcomes ( fewer if less exist) |
| TO-006 | Get outcomes by taxonomy match | get_by_taxonomy({"category": "python"}) | Filtered list of outcomes where taxonomy contains "python" |
| TO-007 | Get outcomes by multi-key taxonomy | get_by_taxonomy({"category": "python", "subcategory": "function"}) | Outcomes matching all specified taxonomy keys |
| TO-008 | Get outcomes by partial taxonomy | get_by_taxonomy({"subcategory": "testing"}) | Outcomes with matching subcategory regardless of category |
| TO-009 | Get statistics for all outcomes | get_statistics() | Dict with: total_count, success_count, failure_count, success_rate |
| TO-010 | Get statistics with taxonomy filter | get_by_taxonomy({"category": "python"}) + get_statistics() | Statistics computed only on filtered outcomes |
| TO-011 | Get empty statistics | Empty store, get_statistics() | Dict with zeros: {total_count: 0, success_count: 0, failure_count: 0, success_rate: 0.0} |
| TO-012 | Get recent with no stored outcomes | get_recent(n=5) on empty store | Empty list [] |
| TO-013 | Get by taxonomy with no matches | get_by_taxonomy({"category": "nonexistent"}) | Empty list [] |
| TO-014 | Record outcome with metadata | TaskOutcome(success=True, metadata={"agent_id": "agent-42", "session": "abc123"}) | Metadata preserved in store |

---

## Test Execution Steps

### TO-001: Record Successful Task Outcome
1. Initialize TaskOutcomeStore instance
2. Create TaskOutcome with success=True and taxonomy data
3. Call `record(outcome)`
4. Verify return value contains outcome_id
5. Call `get_recent(n=1)`
6. Verify returned outcome matches recorded data

### TO-002: Record Failed Task Outcome
1. Initialize TaskOutcomeStore instance
2. Create TaskOutcome with success=False and error_message
3. Call `record(outcome)`
4. Call `get_recent(n=1)`
5. Verify outcome.success == False
6. Verify outcome.error_message matches input

### TO-003: Record Outcome with Context Data
1. Initialize TaskOutcomeStore instance
2. Create TaskOutcome with context dict containing multiple keys
3. Record outcome
4. Retrieve and verify context data is unchanged

### TO-004 & TO-005: Retrieve Recent Outcomes
1. Record 7 TaskOutcome objects with distinct timestamps (or use time.sleep between recordings)
2. Call `get_recent(n=5)`
3. Verify returned list has exactly 5 items
4. Verify items are sorted by timestamp descending (most recent first)
5. Call `get_recent(n=10)`
6. Verify returned list has at most 10 items

### TO-006 & TO-007 & TO-008: Get by Taxonomy
1. Record outcomes with varying taxonomy values
2. Call `get_by_taxonomy(filter_dict)`
3. Verify all returned outcomes match the taxonomy filter
4. For multi-key filters, verify all keys match

### TO-009 & TO-010: Get Statistics
1. Record known number of successful and failed outcomes
2. Call `get_statistics()`
3. Verify total_count matches total recorded
4. Verify success_count and failure_count are accurate
5. Verify success_rate = success_count / total_count

### TO-011 & TO-012 & TO-013: Edge Cases
1. Test on store with no recorded outcomes
2. Verify get_statistics() returns all zeros
3. Verify get_recent() returns empty list
4. Verify get_by_taxonomy() returns empty list for non-matching filter

### TO-014: Record with Metadata
1. Create outcome with metadata dict
2. Record and retrieve
3. Verify metadata is preserved exactly

---

## Pass/Fail Criteria

- **All tests pass** if:
  - All record() calls successfully store outcomes and return valid outcome IDs
  - All get_recent() calls return outcomes sorted by timestamp descending
  - All get_by_taxonomy() calls correctly filter outcomes by taxonomy keys
  - All get_statistics() calls return accurate counts and calculate success_rate correctly
  - Empty store edge cases return appropriate empty/default values
  - Context and metadata are preserved exactly as recorded

- **Test fails** if:
  - Outcomes are not stored or retrieved correctly
  - Sorting order is incorrect
  - Filtering produces false positives or false negatives
  - Statistics calculations are incorrect
  - Data loss occurs with context/metadata fields
  - Exception raised for any edge case input
