#ifndef Py_CPYTHON_COMPILE_H
#  error "this header file must not be included directly"
#endif

/* Public interface */
#define PyCF_MASK (CO_FUTURE_DIVISION | CO_FUTURE_ABSOLUTE_IMPORT | \
                   CO_FUTURE_WITH_STATEMENT | CO_FUTURE_PRINT_FUNCTION | \
                   CO_FUTURE_UNICODE_LITERALS | CO_FUTURE_BARRY_AS_BDFL | \
                   CO_FUTURE_GENERATOR_STOP | CO_FUTURE_ANNOTATIONS)
#define PyCF_MASK_OBSOLETE (CO_NESTED)

/* bpo-39562: CO_FUTURE_ and PyCF_ constants must be kept unique.
   PyCF_ constants can use bits from 0x0100 to 0x10000.
   CO_FUTURE_ constants use bits starting at 0x20000. */
#define PyCF_SOURCE_IS_UTF8  0x0100
#define PyCF_DONT_IMPLY_DEDENT 0x0200
#define PyCF_ONLY_AST 0x0400
#define PyCF_IGNORE_COOKIE 0x0800
#define PyCF_TYPE_COMMENTS 0x1000
#define PyCF_ALLOW_TOP_LEVEL_AWAIT 0x2000
#define PyCF_ALLOW_INCOMPLETE_INPUT 0x4000
#define PyCF_COMPILE_MASK (PyCF_ONLY_AST | PyCF_ALLOW_TOP_LEVEL_AWAIT | \
                           PyCF_TYPE_COMMENTS | PyCF_DONT_IMPLY_DEDENT | \
                           PyCF_ALLOW_INCOMPLETE_INPUT)

typedef struct {
    int cf_flags;  /* bitmask of CO_xxx flags relevant to future */
    int cf_feature_version;  /* minor Python version (PyCF_ONLY_AST) */
} PyCompilerFlags;

#define _PyCompilerFlags_INIT \
    (PyCompilerFlags){.cf_flags = 0, .cf_feature_version = PY_MINOR_VERSION}

/* Future feature support */

typedef struct {
    int ff_features;      /* flags set by future statements */
    int ff_lineno;        /* line number of last future statement */
} PyFutureFeatures;

#define FUTURE_NESTED_SCOPES "nested_scopes"
#define FUTURE_GENERATORS "generators"
#define FUTURE_DIVISION "division"
#define FUTURE_ABSOLUTE_IMPORT "absolute_import"
#define FUTURE_WITH_STATEMENT "with_statement"
#define FUTURE_PRINT_FUNCTION "print_function"
#define FUTURE_UNICODE_LITERALS "unicode_literals"
#define FUTURE_BARRY_AS_BDFL "barry_as_FLUFL"
#define FUTURE_GENERATOR_STOP "generator_stop"
#define FUTURE_ANNOTATIONS "annotations"

#define PY_INVALID_STACK_EFFECT INT_MAX
PyAPI_FUNC(int) PyCompile_OpcodeStackEffect(int opcode, int oparg);
PyAPI_FUNC(int) PyCompile_OpcodeStackEffectWithJump(int opcode, int oparg, int jump);
