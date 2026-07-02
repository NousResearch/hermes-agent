#ifndef Py_CPYTHON_OBJECT_H
#  error "this header file must not be included directly"
#endif

PyAPI_FUNC(void) _Py_NewReference(PyObject *op);

#ifdef Py_TRACE_REFS
/* Py_TRACE_REFS is such major surgery that we call external routines. */
PyAPI_FUNC(void) _Py_ForgetReference(PyObject *);
#endif

#ifdef Py_REF_DEBUG
PyAPI_FUNC(Py_ssize_t) _Py_GetRefTotal(void);
#endif


/********************* String Literals ****************************************/
/* This structure helps managing static strings. The basic usage goes like this:
   Instead of doing

       r = PyObject_CallMethod(o, "foo", "args", ...);

   do

       _Py_IDENTIFIER(foo);
       ...
       r = _PyObject_CallMethodId(o, &PyId_foo, "args", ...);

   PyId_foo is a static variable, either on block level or file level. On first
   usage, the string "foo" is interned, and the structures are linked. On interpreter
   shutdown, all strings are released.

   Alternatively, _Py_static_string allows choosing the variable name.
   _PyUnicode_FromId returns a borrowed reference to the interned string.
   _PyObject_{Get,Set,Has}AttrId are __getattr__ versions using _Py_Identifier*.
*/
typedef struct _Py_Identifier {
    const char* string;
    // Index in PyInterpreterState.unicode.ids.array. It is process-wide
    // unique and must be initialized to -1.
    Py_ssize_t index;
} _Py_Identifier;

#if defined(NEEDS_PY_IDENTIFIER) || !defined(Py_BUILD_CORE)
// For now we are keeping _Py_IDENTIFIER for continued use
// in non-builtin extensions (and naughty PyPI modules).

#define _Py_static_string_init(value) { .string = value, .index = -1 }
#define _Py_static_string(varname, value)  static _Py_Identifier varname = _Py_static_string_init(value)
#define _Py_IDENTIFIER(varname) _Py_static_string(PyId_##varname, #varname)

#endif  /* NEEDS_PY_IDENTIFIER */

typedef int (*getbufferproc)(PyObject *, Py_buffer *, int);
typedef void (*releasebufferproc)(PyObject *, Py_buffer *);

typedef PyObject *(*vectorcallfunc)(PyObject *callable, PyObject *const *args,
                                    size_t nargsf, PyObject *kwnames);


typedef struct {
    /* Number implementations must check *both*
       arguments for proper type and implement the necessary conversions
       in the slot functions themselves. */

    binaryfunc nb_add;
    binaryfunc nb_subtract;
    binaryfunc nb_multiply;
    binaryfunc nb_remainder;
    binaryfunc nb_divmod;
    ternaryfunc nb_power;
    unaryfunc nb_negative;
    unaryfunc nb_positive;
    unaryfunc nb_absolute;
    inquiry nb_bool;
    unaryfunc nb_invert;
    binaryfunc nb_lshift;
    binaryfunc nb_rshift;
    binaryfunc nb_and;
    binaryfunc nb_xor;
    binaryfunc nb_or;
    unaryfunc nb_int;
    void *nb_reserved;  /* the slot formerly known as nb_long */
    unaryfunc nb_float;

    binaryfunc nb_inplace_add;
    binaryfunc nb_inplace_subtract;
    binaryfunc nb_inplace_multiply;
    binaryfunc nb_inplace_remainder;
    ternaryfunc nb_inplace_power;
    binaryfunc nb_inplace_lshift;
    binaryfunc nb_inplace_rshift;
    binaryfunc nb_inplace_and;
    binaryfunc nb_inplace_xor;
    binaryfunc nb_inplace_or;

    binaryfunc nb_floor_divide;
    binaryfunc nb_true_divide;
    binaryfunc nb_inplace_floor_divide;
    binaryfunc nb_inplace_true_divide;

    unaryfunc nb_index;

    binaryfunc nb_matrix_multiply;
    binaryfunc nb_inplace_matrix_multiply;
} PyNumberMethods;

typedef struct {
    lenfunc sq_length;
    binaryfunc sq_concat;
    ssizeargfunc sq_repeat;
    ssizeargfunc sq_item;
    void *was_sq_slice;
    ssizeobjargproc sq_ass_item;
    void *was_sq_ass_slice;
    objobjproc sq_contains;

    binaryfunc sq_inplace_concat;
    ssizeargfunc sq_inplace_repeat;
} PySequenceMethods;

typedef struct {
    lenfunc mp_length;
    binaryfunc mp_subscript;
    objobjargproc mp_ass_subscript;
} PyMappingMethods;

typedef PySendResult (*sendfunc)(PyObject *iter, PyObject *value, PyObject **result);

typedef struct {
    unaryfunc am_await;
    unaryfunc am_aiter;
    unaryfunc am_anext;
    sendfunc am_send;
} PyAsyncMethods;

typedef struct {
     getbufferproc bf_getbuffer;
     releasebufferproc bf_releasebuffer;
} PyBufferProcs;

/* Allow printfunc in the tp_vectorcall_offset slot for
 * backwards-compatibility */
typedef Py_ssize_t printfunc;

// If this structure is modified, Doc/includes/typestruct.h should be updated
// as well.
struct _typeobject {
    PyObject_VAR_HEAD
    const char *tp_name; /* For printing, in format "<module>.<name>" */
    Py_ssize_t tp_basicsize, tp_itemsize; /* For allocation */

    /* Methods to implement standard operations */

    destructor tp_dealloc;
    Py_ssize_t tp_vectorcall_offset;
    getattrfunc tp_getattr;
    setattrfunc tp_setattr;
    PyAsyncMethods *tp_as_async; /* formerly known as tp_compare (Python 2)
                                    or tp_reserved (Python 3) */
    reprfunc tp_repr;

    /* Method suites for standard classes */

    PyNumberMethods *tp_as_number;
    PySequenceMethods *tp_as_sequence;
    PyMappingMethods *tp_as_mapping;

    /* More standard operations (here for binary compatibility) */

    hashfunc tp_hash;
    ternaryfunc tp_call;
    reprfunc tp_str;
    getattrofunc tp_getattro;
    setattrofunc tp_setattro;

    /* Functions to access object as input/output buffer */
    PyBufferProcs *tp_as_buffer;

    /* Flags to define presence of optional/expanded features */
    unsigned long tp_flags;

    const char *tp_doc; /* Documentation string */

    /* Assigned meaning in release 2.0 */
    /* call function for all accessible objects */
    traverseproc tp_traverse;

    /* delete references to contained objects */
    inquiry tp_clear;

    /* Assigned meaning in release 2.1 */
    /* rich comparisons */
    richcmpfunc tp_richcompare;

    /* weak reference enabler */
    Py_ssize_t tp_weaklistoffset;

    /* Iterators */
    getiterfunc tp_iter;
    iternextfunc tp_iternext;

    /* Attribute descriptor and subclassing stuff */
    PyMethodDef *tp_methods;
    PyMemberDef *tp_members;
    PyGetSetDef *tp_getset;
    // Strong reference on a heap type, borrowed reference on a static type
    PyTypeObject *tp_base;
    PyObject *tp_dict;
    descrgetfunc tp_descr_get;
    descrsetfunc tp_descr_set;
    Py_ssize_t tp_dictoffset;
    initproc tp_init;
    allocfunc tp_alloc;
    newfunc tp_new;
    freefunc tp_free; /* Low-level free-memory routine */
    inquiry tp_is_gc; /* For PyObject_IS_GC */
    PyObject *tp_bases;
    PyObject *tp_mro; /* method resolution order */
    PyObject *tp_cache;
    PyObject *tp_subclasses;
    PyObject *tp_weaklist;
    destructor tp_del;

    /* Type attribute cache version tag. Added in version 2.6 */
    unsigned int tp_version_tag;

    destructor tp_finalize;
    vectorcallfunc tp_vectorcall;
};

/* This struct is used by the specializer
 * It should should be treated as an opaque blob
 * by code other than the specializer and interpreter. */
struct _specialization_cache {
    PyObject *getitem;
};

/* The *real* layout of a type object when allocated on the heap */
typedef struct _heaptypeobject {
    /* Note: there's a dependency on the order of these members
       in slotptr() in typeobject.c . */
    PyTypeObject ht_type;
    PyAsyncMethods as_async;
    PyNumberMethods as_number;
    PyMappingMethods as_mapping;
    PySequenceMethods as_sequence; /* as_sequence comes after as_mapping,
                                      so that the mapping wins when both
                                      the mapping and the sequence define
                                      a given operator (e.g. __getitem__).
                                      see add_operators() in typeobject.c . */
    PyBufferProcs as_buffer;
    PyObject *ht_name, *ht_slots, *ht_qualname;
    struct _dictkeysobject *ht_cached_keys;
    PyObject *ht_module;
    char *_ht_tpname;  // Storage for "tp_name"; see PyType_FromModuleAndSpec
    struct _specialization_cache _spec_cache; // For use by the specializer.
    /* here are optional user slots, followed by the members. */
} PyHeapTypeObject;

PyAPI_FUNC(const char *) _PyType_Name(PyTypeObject *);
PyAPI_FUNC(PyObject *) _PyType_Lookup(PyTypeObject *, PyObject *);
PyAPI_FUNC(PyObject *) _PyType_LookupId(PyTypeObject *, _Py_Identifier *);
PyAPI_FUNC(PyObject *) _PyObject_LookupSpecialId(PyObject *, _Py_Identifier *);
#ifndef Py_BUILD_CORE
// Backward compatibility for 3rd-party extensions
// that may be using the old name.
#define _PyObject_LookupSpecial _PyObject_LookupSpecialId
#endif
PyAPI_FUNC(PyTypeObject *) _PyType_CalculateMetaclass(PyTypeObject *, PyObject *);
PyAPI_FUNC(PyObject *) _PyType_GetDocFromInternalDoc(const char *, const char *);
PyAPI_FUNC(PyObject *) _PyType_GetTextSignatureFromInternalDoc(const char *, const char *);
PyAPI_FUNC(PyObject *) PyType_GetModuleByDef(PyTypeObject *, PyModuleDef *);

PyAPI_FUNC(int) PyObject_Print(PyObject *, FILE *, int);
PyAPI_FUNC(void) _Py_BreakPoint(void);
PyAPI_FUNC(void) _PyObject_Dump(PyObject *);
PyAPI_FUNC(int) _PyObject_IsFreed(PyObject *);

PyAPI_FUNC(int) _PyObject_IsAbstract(PyObject *);
PyAPI_FUNC(PyObject *) _PyObject_GetAttrId(PyObject *, _Py_Identifier *);
PyAPI_FUNC(int) _PyObject_SetAttrId(PyObject *, _Py_Identifier *, PyObject *);
/* Replacements of PyObject_GetAttr() and _PyObject_GetAttrId() which
   don't raise AttributeError.

   Return 1 and set *result != NULL if an attribute is found.
   Return 0 and set *result == NULL if an attribute is not found;
   an AttributeError is silenced.
   Return -1 and set *result == NULL if an error other than AttributeError
   is raised.
*/
PyAPI_FUNC(int) _PyObject_LookupAttr(PyObject *, PyObject *, PyObject **);
PyAPI_FUNC(int) _PyObject_LookupAttrId(PyObject *, _Py_Identifier *, PyObject **);

PyAPI_FUNC(int) _PyObject_GetMethod(PyObject *obj, PyObject *name, PyObject **method);

PyAPI_FUNC(PyObject **) _PyObject_GetDictPtr(PyObject *);
PyAPI_FUNC(PyObject *) _PyObject_NextNotImplemented(PyObject *);
PyAPI_FUNC(void) PyObject_CallFinalizer(PyObject *);
PyAPI_FUNC(int) PyObject_CallFinalizerFromDealloc(PyObject *);

/* Same as PyObject_Generic{Get,Set}Attr, but passing the attributes
   dict as the last parameter. */
PyAPI_FUNC(PyObject *)
_PyObject_GenericGetAttrWithDict(PyObject *, PyObject *, PyObject *, int);
PyAPI_FUNC(int)
_PyObject_GenericSetAttrWithDict(PyObject *, PyObject *,
                                 PyObject *, PyObject *);

PyAPI_FUNC(PyObject *) _PyObject_FunctionStr(PyObject *);

/* Safely decref `op` and set `op` to `op2`.
 *
 * As in case of Py_CLEAR "the obvious" code can be deadly:
 *
 *     Py_DECREF(op);
 *     op = op2;
 *
 * The safe way is:
 *
 *      Py_SETREF(op, op2);
 *
 * That arranges to set `op` to `op2` _before_ decref'ing, so that any code
 * triggered as a side-effect of `op` getting torn down no longer believes
 * `op` points to a valid object.
 *
 * Py_XSETREF is a variant of Py_SETREF that uses Py_XDECREF instead of
 * Py_DECREF.
 */

#define Py_SETREF(op, op2)                      \
    do {                                        \
        PyObject *_py_tmp = _PyObject_CAST(op); \
        (op) = (op2);                           \
        Py_DECREF(_py_tmp);                     \
    } while (0)

#define Py_XSETREF(op, op2)                     \
    do {                                        \
        PyObject *_py_tmp = _PyObject_CAST(op); \
        (op) = (op2);                           \
        Py_XDECREF(_py_tmp);                    \
    } while (0)


PyAPI_DATA(PyTypeObject) _PyNone_Type;
PyAPI_DATA(PyTypeObject) _PyNotImplemented_Type;

/* Maps Py_LT to Py_GT, ..., Py_GE to Py_LE.
 * Defined in object.c.
 */
PyAPI_DATA(int) _Py_SwappedOp[];

PyAPI_FUNC(void)
_PyDebugAllocatorStats(FILE *out, const char *block_name, int num_blocks,
                       size_t sizeof_block);
PyAPI_FUNC(void)
_PyObject_DebugTypeStats(FILE *out);

/* Define a pair of assertion macros:
   _PyObject_ASSERT_FROM(), _PyObject_ASSERT_WITH_MSG() and _PyObject_ASSERT().

   These work like the regular C assert(), in that they will abort the
   process with a message on stderr if the given condition fails to hold,
   but compile away to nothing if NDEBUG is defined.

   However, before aborting, Python will also try to call _PyObject_Dump() on
   the given object.  This may be of use when investigating bugs in which a
   particular object is corrupt (e.g. buggy a tp_visit method in an extension
   module breaking the garbage collector), to help locate the broken objects.

   The WITH_MSG variant allows you to supply an additional message that Python
   will attempt to print to stderr, after the object dump. */
#ifdef NDEBUG
   /* No debugging: compile away the assertions: */
#  define _PyObject_ASSERT_FROM(obj, expr, msg, filename, lineno, func) \
    ((void)0)
#else
   /* With debugging: generate checks: */
#  define _PyObject_ASSERT_FROM(obj, expr, msg, filename, lineno, func) \
    ((expr) \
      ? (void)(0) \
      : _PyObject_AssertFailed((obj), Py_STRINGIFY(expr), \
                               (msg), (filename), (lineno), (func)))
#endif

#define _PyObject_ASSERT_WITH_MSG(obj, expr, msg) \
    _PyObject_ASSERT_FROM(obj, expr, msg, __FILE__, __LINE__, __func__)
#define _PyObject_ASSERT(obj, expr) \
    _PyObject_ASSERT_WITH_MSG(obj, expr, NULL)

#define _PyObject_ASSERT_FAILED_MSG(obj, msg) \
    _PyObject_AssertFailed((obj), NULL, (msg), __FILE__, __LINE__, __func__)

/* Declare and define _PyObject_AssertFailed() even when NDEBUG is defined,
   to avoid causing compiler/linker errors when building extensions without
   NDEBUG against a Python built with NDEBUG defined.

   msg, expr and function can be NULL. */
PyAPI_FUNC(void) _Py_NO_RETURN _PyObject_AssertFailed(
    PyObject *obj,
    const char *expr,
    const char *msg,
    const char *file,
    int line,
    const char *function);

/* Check if an object is consistent. For example, ensure that the reference
   counter is greater than or equal to 1, and ensure that ob_type is not NULL.

   Call _PyObject_AssertFailed() if the object is inconsistent.

   If check_content is zero, only check header fields: reduce the overhead.

   The function always return 1. The return value is just here to be able to
   write:

   assert(_PyObject_CheckConsistency(obj, 1)); */
PyAPI_FUNC(int) _PyObject_CheckConsistency(
    PyObject *op,
    int check_content);


/* Trashcan mechanism, thanks to Christian Tismer.

When deallocating a container object, it's possible to trigger an unbounded
chain of deallocations, as each Py_DECREF in turn drops the refcount on "the
next" object in the chain to 0.  This can easily lead to stack overflows,
especially in threads (which typically have less stack space to work with).

A container object can avoid this by bracketing the body of its tp_dealloc
function with a pair of macros:

static void
mytype_dealloc(mytype *p)
{
    ... declarations go here ...

    PyObject_GC_UnTrack(p);        // must untrack first
    Py_TRASHCAN_BEGIN(p, mytype_dealloc)
    ... The body of the deallocator goes here, including all calls ...
    ... to Py_DECREF on contained objects.                         ...
    Py_TRASHCAN_END                // there should be no code after this
}

CAUTION:  Never return from the middle of the body!  If the body needs to
"get out early", put a label immediately before the Py_TRASHCAN_END
call, and goto it.  Else the call-depth counter (see below) will stay
above 0 forever, and the trashcan will never get emptied.

How it works:  The BEGIN macro increments a call-depth counter.  So long
as this counter is small, the body of the deallocator is run directly without
further ado.  But if the counter gets large, it instead adds p to a list of
objects to be deallocated later, skips the body of the deallocator, and
resumes execution after the END macro.  The tp_dealloc routine then returns
without deallocating anything (and so unbounded call-stack depth is avoided).

When the call stack finishes unwinding again, code generated by the END macro
notices this, and calls another routine to deallocate all the objects that
may have been added to the list of deferred deallocations.  In effect, a
chain of N deallocations is broken into (N-1)/(_PyTrash_UNWIND_LEVEL-1) pieces,
with the call stack never exceeding a depth of _PyTrash_UNWIND_LEVEL.

Since the tp_dealloc of a subclass typically calls the tp_dealloc of the base
class, we need to ensure that the trashcan is only triggered on the tp_dealloc
of the actual class being deallocated. Otherwise we might end up with a
partially-deallocated object. To check this, the tp_dealloc function must be
passed as second argument to Py_TRASHCAN_BEGIN().
*/

/* Python 3.9 private API, invoked by the macros below. */
PyAPI_FUNC(int) _PyTrash_begin(PyThreadState *tstate, PyObject *op);
PyAPI_FUNC(void) _PyTrash_end(PyThreadState *tstate);
/* Python 3.10 private API, invoked by the Py_TRASHCAN_BEGIN(). */
PyAPI_FUNC(int) _PyTrash_cond(PyObject *op, destructor dealloc);

#define Py_TRASHCAN_BEGIN_CONDITION(op, cond) \
    do { \
        PyThreadState *_tstate = NULL; \
        /* If "cond" is false, then _tstate remains NULL and the deallocator \
         * is run normally without involving the trashcan */ \
        if (cond) { \
            _tstate = PyThreadState_Get(); \
            if (_PyTrash_begin(_tstate, _PyObject_CAST(op))) { \
                break; \
            } \
        }
        /* The body of the deallocator is here. */
#define Py_TRASHCAN_END \
        if (_tstate) { \
            _PyTrash_end(_tstate); \
        } \
    } while (0);

#define Py_TRASHCAN_BEGIN(op, dealloc) \
    Py_TRASHCAN_BEGIN_CONDITION(op, \
        _PyTrash_cond(_PyObject_CAST(op), (destructor)dealloc))

/* The following two macros, Py_TRASHCAN_SAFE_BEGIN and
 * Py_TRASHCAN_SAFE_END, are deprecated since version 3.11 and
 * will be removed in the future.
 * Use Py_TRASHCAN_BEGIN and Py_TRASHCAN_END instead.
 */
Py_DEPRECATED(3.11) typedef int UsingDeprecatedTrashcanMacro;
#define Py_TRASHCAN_SAFE_BEGIN(op) \
    do { \
        UsingDeprecatedTrashcanMacro cond=1; \
        Py_TRASHCAN_BEGIN_CONDITION(op, cond);
#define Py_TRASHCAN_SAFE_END(op) \
        Py_TRASHCAN_END; \
    } while(0);
