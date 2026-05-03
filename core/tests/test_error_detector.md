# Test Documentation: ErrorPatternDetector

## Overview

This document provides test cases for the `ErrorPatternDetector` class. This component detects error patterns from logs or error lists, normalizes errors for comparison, calculates similarity between errors, and infers root causes from error clusters.

---

## Test Case Table

| ID | Description | Input | Expected Output |
|----|-------------|-------|-----------------|
| ED-001 | Detect known error pattern | detect(["FileNotFoundError: [Errno 2] No such file: 'config.yaml'"]) | List with matched pattern including pattern_id, description, and match_score |
| ED-002 | Detect multiple errors | detect(["TypeError", "ValueError: invalid literal"]) | List of detected patterns for each error |
| ED-003 | Detect no errors | detect([]) | Empty list [] |
| ED-004 | Detect empty string in list | detect([""]) | Empty list [] or list with unknown pattern |
| ED-005 | Normalize Python traceback error | _normalize_error("TypeError: unsupported operand type(s) for +: 'int' and 'str'") | Normalized string with error type preserved |
| ED-006 | Normalize error with file path | _normalize_error("Error at /home/user/project/main.py:42") | Path stripped, core error preserved |
| ED-007 | Normalize error with line number | _normalize_error("SyntaxError: invalid syntax at line 15") | Line number stripped, error type and message preserved |
| ED-008 | Normalize empty string | _normalize_error("") | Empty string |
| ED-009 | Calculate similarity for identical errors | _similarity("ConnectionRefusedError", "ConnectionRefusedError") | 1.0 (maximum similarity) |
| ED-010 | Calculate similarity for same error type | _similarity("TypeError: a", "TypeError: b") | High score (>0.7) |
| ED-011 | Calculate similarity for different errors | _similarity("TypeError", "FileNotFoundError") | Low score (<0.3) |
| ED-012 | Calculate similarity with empty strings | _similarity("", "") | 1.0 or handled gracefully |
| ED-013 | Infer root cause for repeated errors | Errors: ["ConnectionRefusedError"] * 5 | Inferred root cause pointing to network/connection issue |
| ED-014 | Infer root cause for varied errors | Errors: ["TypeError", "ValueError", "KeyError"] | Generic root cause or multiple root causes identified |
| ED-015 | Infer root cause for single error | Errors: ["MemoryError"] | Specific root cause: memory/resource exhaustion |
| ED-016 | Infer root cause for empty list | _infer_root_cause([]) | Empty result or null/None |
| ED-017 | Similar errors with different messages | detect(["Error: timeout", "Error: connection reset"]) | Clustered as similar error patterns |
| ED-018 | Different error categories | detect(["ImportError", "SyntaxError", "RuntimeError"]) | Distinct patterns identified separately |
| ED-019 | Detect with custom patterns | detect(["CustomError: custom message"], custom_patterns=[...]) | Custom pattern matched if defined |
| ED-020 | Detect similar errors with thresholds | Errors with 0.85 similarity | Grouped as same pattern family |

---

## Test Execution Steps

### ED-001 & ED-002: Basic Detection
1. Initialize ErrorPatternDetector with default patterns
2. Call detect() with list containing known error strings
3. Verify return is a list of DetectionResult objects
4. For ED-002, verify both errors are detected

### ED-003 & ED-004: Empty List Handling
1. Call detect() with empty list
2. Verify empty list returned
3. Call detect() with list containing empty string
4. Verify graceful handling (empty string filtered out)

### ED-005 & ED-006 & ED-007: Normalize Error
1. Initialize ErrorPatternDetector
2. Call _normalize_error() with various error formats
3. Verify normalized output:
   - Error type (e.g., TypeError) is preserved
   - File paths are removed or replaced with placeholder
   - Line numbers are removed
   - Core error message is retained

### ED-008: Normalize Empty String
1. Call _normalize_error("")
2. Verify returns empty string without exception

### ED-009 & ED-010 & ED-011: Similarity Calculation
1. Call _similarity() with pairs of errors
2. Verify identical errors return highest score (1.0)
3. Verify same error type with different messages returns high score
4. Verify completely different errors return low score

### ED-012: Similarity with Empty Strings
1. Call _similarity("", "")
2. Verify handled gracefully (returns 1.0 or defined behavior)

### ED-013 & ED-014 & ED-015: Root Cause Inference
1. Call _infer_root_cause() with list of errors
2. For repeated similar errors (ED-013), verify specific root cause identified
3. For varied errors (ED-014), verify general or multiple causes identified
4. For single error (ED-015), verify specific cause for that error type

### ED-016: Root Cause Empty List
1. Call _infer_root_cause([])
2. Verify returns empty/null result without exception

### ED-017 & ED-018: Similar vs Different Errors
1. Prepare error lists with similar errors (same type, different message)
2. Call detect() and verify errors are clustered/grouped appropriately
3. Prepare error list with distinct error types
4. Verify each type is identified separately

### ED-019: Custom Patterns
1. Define custom error patterns
2. Call detect() with custom_patterns parameter
3. Verify custom patterns are matched

### ED-020: Similarity Threshold
1. Create errors with controlled similarity (~0.85)
2. Call detect() with appropriate threshold
3. Verify grouping behavior matches threshold

---

## Pass/Fail Criteria

- **All tests pass** if:
  - detect() returns list of DetectionResult objects for valid errors
  - detect() returns empty list for empty input
  - _normalize_error() consistently strips paths and line numbers while preserving error type
  - _similarity() returns 1.0 for identical errors
  - _similarity() returns high score (>0.7) for same error type
  - _similarity() returns low score (<0.3) for unrelated errors
  - _infer_root_cause() returns meaningful cause for repeated errors
  - _infer_root_cause() handles empty list gracefully
  - Custom patterns are matched when provided
  - Similar errors are grouped correctly based on threshold

- **Test fails** if:
  - detect() raises exception for valid input
  - Normalization is inconsistent or loses critical error information
  - Similarity scores are counterintuitive (e.g., identical errors score low)
  - Root cause inference produces incorrect or misleading results
  - Empty inputs cause exceptions instead of graceful handling
