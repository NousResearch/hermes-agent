/* Definitions for bytecode */

#ifndef Py_LIMITED_API
#ifndef Py_CODE_H
#define Py_CODE_H
#ifdef __cplusplus
extern "C" {
#endif

/* Each instruction in a code object is a fixed-width value,
 * currently 2 bytes: 1-byte opcode + 1-byte oparg.  The EXTENDED_ARG
 * opcode allows for larger values but the current limit is 3 uses
 * of EXTENDED_ARG (see Python/compile.c), for a maximum
 * 32-bit value.  This aligns with the note in Python/compile.c
 * (compiler_addop_i_line) indicating that the max oparg value is
 * 2**32 - 1, rather than INT_MAX.
 */

typedef uint16_t _Py_CODEUNIT;

#ifdef WORDS_BIGENDIAN
#  define _Py_OPCODE(word) ((word) >> 8)
#  define _Py_OPARG(word) ((word) & 255)
#  define _Py_MAKECODEUNIT(opcode, oparg) (((opcode)<<8)|(oparg))
#else
#  define _Py_OPCODE(word) ((word) & 255)
#  define _Py_OPARG(word) ((word) >> 8)
#  define _Py_MAKECODEUNIT(opcode, oparg) ((opcode)|((oparg)<<8))
#endif

// Use "unsigned char" instead of "uint8_t" here to avoid illegal aliasing:
#define _Py_SET_OPCODE(word, opcode) (((unsigned char *)&(word))[0] = (opcode))

// To avoid repeating ourselves in deepfreeze.py, all PyCodeObject members are
// defined in this macro:
#define _PyCode_DEF(SIZE) {                                                    \
    PyObject_VAR_HEAD                                                          \
                                                                               \
    /* Note only the following fields are used in hash and/or comparisons      \
     *                                                                         \
     * - co_name                                                               \
     * - co_argcount                                                           \
     * - co_posonlyargcount                                                    \
     * - co_kwonlyargcount                                                     \
     * - co_nlocals                                                            \
     * - co_stacksize                                                          \
     * - co_flags                                                              \
     * - co_firstlineno                                                        \
     * - co_consts                                                             \
     * - co_names                                                              \
     * - co_localsplusnames                                                    \
     * This is done to preserve the name and line number for tracebacks        \
     * and debuggers; otherwise, constant de-duplication would collapse        \
     * identical functions/lambdas defined on different lines.                 \
     */                                                                        \
                                                                               \
    /* These fields are set with provided values on new code objects. */       \
                                                                               \
    /* The hottest fields (in the eval loop) are grouped here at the top. */   \
    PyObject *co_consts;           /* list (constants used) */                 \
    PyObject *co_names;            /* list of strings (names used) */          \
    PyObject *co_exceptiontable;   /* Byte string encoding exception handling  \
                                      table */                                 \
    int co_flags;                  /* CO_..., see below */                     \
    short co_warmup;                 /* Warmup counter for quickening */       \
    short _co_linearray_entry_size;  /* Size of each entry in _co_linearray */ \
                                                                               \
    /* The rest are not so impactful on performance. */                        \
    int co_argcount;              /* #arguments, except *args */               \
    int co_posonlyargcount;       /* #positional only arguments */             \
    int co_kwonlyargcount;        /* #keyword only arguments */                \
    int co_stacksize;             /* #entries needed for evaluation stack */   \
    int co_firstlineno;           /* first source line number */               \
                                                                               \
    /* redundant values (derived from co_localsplusnames and                   \
       co_localspluskinds) */                                                  \
    int co_nlocalsplus;           /* number of local + cell + free variables   \
                                  */                                           \
    int co_nlocals;               /* number of local variables */              \
    int co_nplaincellvars;        /* number of non-arg cell variables */       \
    int co_ncellvars;             /* total number of cell variables */         \
    int co_nfreevars;             /* number of free variables */               \
                                                                               \
    PyObject *co_localsplusnames; /* tuple mapping offsets to names */         \
    PyObject *co_localspluskinds; /* Bytes mapping to local kinds (one byte    \
                                     per variable) */                          \
    PyObject *co_filename;        /* unicode (where it was loaded from) */     \
    PyObject *co_name;            /* unicode (name, for reference) */          \
    PyObject *co_qualname;        /* unicode (qualname, for reference) */      \
    PyObject *co_linetable;       /* bytes object that holds location info */  \
    PyObject *co_weakreflist;     /* to support weakrefs to code objects */    \
    PyObject *_co_code;           /* cached co_code object/attribute */        \
    char *_co_linearray;          /* array of line offsets */                  \
    int _co_firsttraceable;       /* index of first traceable instruction */   \
    /* Scratch space for extra data relating to the code object.               \
       Type is a void* to keep the format private in codeobject.c to force     \
       people to go through the proper APIs. */                                \
    void *co_extra;                                                            \
    char co_code_adaptive[(SIZE)];                                             \
}

/* Bytecode object */
struct PyCodeObject _PyCode_DEF(1);

/* Masks for co_flags above */
#define CO_OPTIMIZED    0x0001
#define CO_NEWLOCALS    0x0002
#define CO_VARARGS      0x0004
#define CO_VARKEYWORDS  0x0008
#define CO_NESTED       0x0010
#define CO_GENERATOR    0x0020

/* The CO_COROUTINE flag is set for coroutine functions (defined with
   ``async def`` keywords) */
#define CO_COROUTINE            0x0080
#define CO_ITERABLE_COROUTINE   0x0100
#define CO_ASYNC_GENERATOR      0x0200

/* bpo-39562: These constant values are changed in Python 3.9
   to prevent collision with compiler flags. CO_FUTURE_ and PyCF_
   constants must be kept unique. PyCF_ constants can use bits from
   0x0100 to 0x10000. CO_FUTURE_ constants use bits starting at 0x20000. */
#define CO_FUTURE_DIVISION      0x20000
#define CO_FUTURE_ABSOLUTE_IMPORT 0x40000 /* do absolute imports by default */
#define CO_FUTURE_WITH_STATEMENT  0x80000
#define CO_FUTURE_PRINT_FUNCTION  0x100000
#define CO_FUTURE_UNICODE_LITERALS 0x200000

#define CO_FUTURE_BARRY_AS_BDFL  0x400000
#define CO_FUTURE_GENERATOR_STOP  0x800000
#define CO_FUTURE_ANNOTATIONS    0x1000000

/* This should be defined if a future statement modifies the syntax.
   For example, when a keyword is added.
*/
#define PY_PARSER_REQUIRES_FUTURE_KEYWORD

#define CO_MAXBLOCKS 20 /* Max static block nesting within a function */

PyAPI_DATA(PyTypeObject) PyCode_Type;

#define PyCode_Check(op) Py_IS_TYPE(op, &PyCode_Type)
#define PyCode_GetNumFree(op) ((op)->co_nfreevars)
#define _PyCode_CODE(CO) ((_Py_CODEUNIT *)(CO)->co_code_adaptive)
#define _PyCode_NBYTES(CO) (Py_SIZE(CO) * (Py_ssize_t)sizeof(_Py_CODEUNIT))

/* Public interface */
PyAPI_FUNC(PyCodeObject *) PyCode_New(
        int, int, int, int, int, PyObject *, PyObject *,
        PyObject *, PyObject *, PyObject *, PyObject *,
        PyObject *, PyObject *, PyObject *, int, PyObject *,
        PyObject *);

PyAPI_FUNC(PyCodeObject *) PyCode_NewWithPosOnlyArgs(
        int, int, int, int, int, int, PyObject *, PyObject *,
        PyObject *, PyObject *, PyObject *, PyObject *,
        PyObject *, PyObject *, PyObject *, int, PyObject *,
        PyObject *);
        /* same as struct above */

/* Creates a new empty code object with the specified source location. */
PyAPI_FUNC(PyCodeObject *)
PyCode_NewEmpty(const char *filename, const char *funcname, int firstlineno);

/* Return the line number associated with the specified bytecode index
   in this code object.  If you just need the line number of a frame,
   use PyFrame_GetLineNumber() instead. */
PyAPI_FUNC(int) PyCode_Addr2Line(PyCodeObject *, int);

PyAPI_FUNC(int) PyCode_Addr2Location(PyCodeObject *, int, int *, int *, int *, int *);

/* for internal use only */
struct _opaque {
    int computed_line;
    const uint8_t *lo_next;
    const uint8_t *limit;
};

typedef struct _line_offsets {
    int ar_start;
    int ar_end;
    int ar_line;
    struct _opaque opaque;
} PyCodeAddressRange;

/* Update *bounds to describe the first and one-past-the-last instructions in the
   same line as lasti.  Return the number of that line.
*/
PyAPI_FUNC(int) _PyCode_CheckLineNumber(int lasti, PyCodeAddressRange *bounds);

/* Create a comparable key used to compare constants taking in account the
 * object type. It is used to make sure types are not coerced (e.g., float and
 * complex) _and_ to distinguish 0.0 from -0.0 e.g. on IEEE platforms
 *
 * Return (type(obj), obj, ...): a tuple with variable size (at least 2 items)
 * depending on the type and the value. The type is the first item to not
 * compare bytes and str which can raise a BytesWarning exception. */
PyAPI_FUNC(PyObject*) _PyCode_ConstantKey(PyObject *obj);

PyAPI_FUNC(PyObject*) PyCode_Optimize(PyObject *code, PyObject* consts,
                                      PyObject *names, PyObject *lnotab);


PyAPI_FUNC(int) _PyCode_GetExtra(PyObject *code, Py_ssize_t index,
                                 void **extra);
PyAPI_FUNC(int) _PyCode_SetExtra(PyObject *code, Py_ssize_t index,
                                 void *extra);

/* Equivalent to getattr(code, 'co_code') in Python.
   Returns a strong reference to a bytes object. */
PyAPI_FUNC(PyObject *) PyCode_GetCode(PyCodeObject *code);
/* Equivalent to getattr(code, 'co_varnames') in Python. */
PyAPI_FUNC(PyObject *) PyCode_GetVarnames(PyCodeObject *code);
/* Equivalent to getattr(code, 'co_cellvars') in Python. */
PyAPI_FUNC(PyObject *) PyCode_GetCellvars(PyCodeObject *code);
/* Equivalent to getattr(code, 'co_freevars') in Python. */
PyAPI_FUNC(PyObject *) PyCode_GetFreevars(PyCodeObject *code);

typedef enum _PyCodeLocationInfoKind {
    /* short forms are 0 to 9 */
    PY_CODE_LOCATION_INFO_SHORT0 = 0,
    /* one lineforms are 10 to 12 */
    PY_CODE_LOCATION_INFO_ONE_LINE0 = 10,
    PY_CODE_LOCATION_INFO_ONE_LINE1 = 11,
    PY_CODE_LOCATION_INFO_ONE_LINE2 = 12,

    PY_CODE_LOCATION_INFO_NO_COLUMNS = 13,
    PY_CODE_LOCATION_INFO_LONG = 14,
    PY_CODE_LOCATION_INFO_NONE = 15
} _PyCodeLocationInfoKind;

#ifdef __cplusplus
}
#endif
#endif  // !Py_CODE_H
#endif  // !Py_LIMITED_API
