#ifndef Py_INTERNAL_OBJECT_H
#define Py_INTERNAL_OBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include <stdbool.h>
#include "pycore_gc.h"            // _PyObject_GC_IS_TRACKED()
#include "pycore_interp.h"        // PyInterpreterState.gc
#include "pycore_pystate.h"       // _PyInterpreterState_GET()
#include "pycore_runtime.h"       // _PyRuntime

#define _PyObject_IMMORTAL_INIT(type) \
    { \
        .ob_refcnt = 999999999, \
        .ob_type = type, \
    }
#define _PyVarObject_IMMORTAL_INIT(type, size) \
    { \
        .ob_base = _PyObject_IMMORTAL_INIT(type), \
        .ob_size = size, \
    }

PyAPI_FUNC(void) _Py_NO_RETURN _Py_FatalRefcountErrorFunc(
    const char *func,
    const char *message);

#define _Py_FatalRefcountError(message) _Py_FatalRefcountErrorFunc(__func__, message)

static inline void
_Py_DECREF_SPECIALIZED(PyObject *op, const destructor destruct)
{
#ifdef Py_REF_DEBUG
    _Py_RefTotal--;
#endif
    if (--op->ob_refcnt != 0) {
        assert(op->ob_refcnt > 0);
    }
    else {
#ifdef Py_TRACE_REFS
        _Py_ForgetReference(op);
#endif
        destruct(op);
    }
}

static inline void
_Py_DECREF_NO_DEALLOC(PyObject *op)
{
#ifdef Py_REF_DEBUG
    _Py_RefTotal--;
#endif
    op->ob_refcnt--;
#ifdef Py_DEBUG
    if (op->ob_refcnt <= 0) {
        _Py_FatalRefcountError("Expected a positive remaining refcount");
    }
#endif
}

PyAPI_FUNC(int) _PyType_CheckConsistency(PyTypeObject *type);
PyAPI_FUNC(int) _PyDict_CheckConsistency(PyObject *mp, int check_content);

/* Update the Python traceback of an object. This function must be called
   when a memory block is reused from a free list.

   Internal function called by _Py_NewReference(). */
extern int _PyTraceMalloc_NewReference(PyObject *op);

// Fast inlined version of PyType_HasFeature()
static inline int
_PyType_HasFeature(PyTypeObject *type, unsigned long feature) {
    return ((type->tp_flags & feature) != 0);
}

extern void _PyType_InitCache(PyInterpreterState *interp);


/* Inline functions trading binary compatibility for speed:
   _PyObject_Init() is the fast version of PyObject_Init(), and
   _PyObject_InitVar() is the fast version of PyObject_InitVar().

   These inline functions must not be called with op=NULL. */
static inline void
_PyObject_Init(PyObject *op, PyTypeObject *typeobj)
{
    assert(op != NULL);
    Py_SET_TYPE(op, typeobj);
    if (_PyType_HasFeature(typeobj, Py_TPFLAGS_HEAPTYPE)) {
        Py_INCREF(typeobj);
    }
    _Py_NewReference(op);
}

static inline void
_PyObject_InitVar(PyVarObject *op, PyTypeObject *typeobj, Py_ssize_t size)
{
    assert(op != NULL);
    Py_SET_SIZE(op, size);
    _PyObject_Init((PyObject *)op, typeobj);
}


/* Tell the GC to track this object.
 *
 * The object must not be tracked by the GC.
 *
 * NB: While the object is tracked by the collector, it must be safe to call the
 * ob_traverse method.
 *
 * Internal note: interp->gc.generation0->_gc_prev doesn't have any bit flags
 * because it's not object header.  So we don't use _PyGCHead_PREV() and
 * _PyGCHead_SET_PREV() for it to avoid unnecessary bitwise operations.
 *
 * See also the public PyObject_GC_Track() function.
 */
static inline void _PyObject_GC_TRACK(
// The preprocessor removes _PyObject_ASSERT_FROM() calls if NDEBUG is defined
#ifndef NDEBUG
    const char *filename, int lineno,
#endif
    PyObject *op)
{
    _PyObject_ASSERT_FROM(op, !_PyObject_GC_IS_TRACKED(op),
                          "object already tracked by the garbage collector",
                          filename, lineno, __func__);

    PyGC_Head *gc = _Py_AS_GC(op);
    _PyObject_ASSERT_FROM(op,
                          (gc->_gc_prev & _PyGC_PREV_MASK_COLLECTING) == 0,
                          "object is in generation which is garbage collected",
                          filename, lineno, __func__);

    PyInterpreterState *interp = _PyInterpreterState_GET();
    PyGC_Head *generation0 = interp->gc.generation0;
    PyGC_Head *last = (PyGC_Head*)(generation0->_gc_prev);
    _PyGCHead_SET_NEXT(last, gc);
    _PyGCHead_SET_PREV(gc, last);
    _PyGCHead_SET_NEXT(gc, generation0);
    generation0->_gc_prev = (uintptr_t)gc;
}

/* Tell the GC to stop tracking this object.
 *
 * Internal note: This may be called while GC. So _PyGC_PREV_MASK_COLLECTING
 * must be cleared. But _PyGC_PREV_MASK_FINALIZED bit is kept.
 *
 * The object must be tracked by the GC.
 *
 * See also the public PyObject_GC_UnTrack() which accept an object which is
 * not tracked.
 */
static inline void _PyObject_GC_UNTRACK(
// The preprocessor removes _PyObject_ASSERT_FROM() calls if NDEBUG is defined
#ifndef NDEBUG
    const char *filename, int lineno,
#endif
    PyObject *op)
{
    _PyObject_ASSERT_FROM(op, _PyObject_GC_IS_TRACKED(op),
                          "object not tracked by the garbage collector",
                          filename, lineno, __func__);

    PyGC_Head *gc = _Py_AS_GC(op);
    PyGC_Head *prev = _PyGCHead_PREV(gc);
    PyGC_Head *next = _PyGCHead_NEXT(gc);
    _PyGCHead_SET_NEXT(prev, next);
    _PyGCHead_SET_PREV(next, prev);
    gc->_gc_next = 0;
    gc->_gc_prev &= _PyGC_PREV_MASK_FINALIZED;
}

// Macros to accept any type for the parameter, and to automatically pass
// the filename and the filename (if NDEBUG is not defined) where the macro
// is called.
#ifdef NDEBUG
#  define _PyObject_GC_TRACK(op) \
        _PyObject_GC_TRACK(_PyObject_CAST(op))
#  define _PyObject_GC_UNTRACK(op) \
        _PyObject_GC_UNTRACK(_PyObject_CAST(op))
#else
#  define _PyObject_GC_TRACK(op) \
        _PyObject_GC_TRACK(__FILE__, __LINE__, _PyObject_CAST(op))
#  define _PyObject_GC_UNTRACK(op) \
        _PyObject_GC_UNTRACK(__FILE__, __LINE__, _PyObject_CAST(op))
#endif

#ifdef Py_REF_DEBUG
extern void _PyDebug_PrintTotalRefs(void);
#endif

#ifdef Py_TRACE_REFS
extern void _Py_AddToAllObjects(PyObject *op, int force);
extern void _Py_PrintReferences(FILE *);
extern void _Py_PrintReferenceAddresses(FILE *);
#endif

static inline PyObject **
_PyObject_GET_WEAKREFS_LISTPTR(PyObject *op)
{
    Py_ssize_t offset = Py_TYPE(op)->tp_weaklistoffset;
    return (PyObject **)((char *)op + offset);
}

// Fast inlined version of PyObject_IS_GC()
static inline int
_PyObject_IS_GC(PyObject *obj)
{
    return (PyType_IS_GC(Py_TYPE(obj))
            && (Py_TYPE(obj)->tp_is_gc == NULL
                || Py_TYPE(obj)->tp_is_gc(obj)));
}

// Fast inlined version of PyType_IS_GC()
#define _PyType_IS_GC(t) _PyType_HasFeature((t), Py_TPFLAGS_HAVE_GC)

static inline size_t
_PyType_PreHeaderSize(PyTypeObject *tp)
{
    return _PyType_IS_GC(tp) * sizeof(PyGC_Head) +
        _PyType_HasFeature(tp, Py_TPFLAGS_MANAGED_DICT) * 2 * sizeof(PyObject *);
}

void _PyObject_GC_Link(PyObject *op);

// Usage: assert(_Py_CheckSlotResult(obj, "__getitem__", result != NULL));
extern int _Py_CheckSlotResult(
    PyObject *obj,
    const char *slot_name,
    int success);

// PyType_Ready() must be called if _PyType_IsReady() is false.
// See also the Py_TPFLAGS_READY flag.
#define _PyType_IsReady(type) ((type)->tp_dict != NULL)

// Test if a type supports weak references
static inline int _PyType_SUPPORTS_WEAKREFS(PyTypeObject *type) {
    return (type->tp_weaklistoffset > 0);
}

extern PyObject* _PyType_AllocNoTrack(PyTypeObject *type, Py_ssize_t nitems);

extern int _PyObject_InitializeDict(PyObject *obj);
extern int _PyObject_StoreInstanceAttribute(PyObject *obj, PyDictValues *values,
                                          PyObject *name, PyObject *value);
PyObject * _PyObject_GetInstanceAttribute(PyObject *obj, PyDictValues *values,
                                        PyObject *name);

static inline PyDictValues **_PyObject_ValuesPointer(PyObject *obj)
{
    assert(Py_TYPE(obj)->tp_flags & Py_TPFLAGS_MANAGED_DICT);
    return ((PyDictValues **)obj)-4;
}

static inline PyObject **_PyObject_ManagedDictPointer(PyObject *obj)
{
    assert(Py_TYPE(obj)->tp_flags & Py_TPFLAGS_MANAGED_DICT);
    return ((PyObject **)obj)-3;
}

#define MANAGED_DICT_OFFSET (((int)sizeof(PyObject *))*-3)

extern PyObject ** _PyObject_DictPointer(PyObject *);
extern int _PyObject_VisitInstanceAttributes(PyObject *self, visitproc visit, void *arg);
extern void _PyObject_ClearInstanceAttributes(PyObject *self);
extern void _PyObject_FreeInstanceAttributes(PyObject *self);
extern int _PyObject_IsInstanceDictEmpty(PyObject *);
extern PyObject* _PyType_GetSubclasses(PyTypeObject *);

// Access macro to the members which are floating "behind" the object
#define _PyHeapType_GET_MEMBERS(etype) \
    ((PyMemberDef *)(((char *)etype) + Py_TYPE(etype)->tp_basicsize))

PyAPI_FUNC(PyObject *) _PyObject_LookupSpecial(PyObject *, PyObject *);

/* C function call trampolines to mitigate bad function pointer casts.
 *
 * Typical native ABIs ignore additional arguments or fill in missing
 * values with 0/NULL in function pointer cast. Compilers do not show
 * warnings when a function pointer is explicitly casted to an
 * incompatible type.
 *
 * Bad fpcasts are an issue in WebAssembly. WASM's indirect_call has strict
 * function signature checks. Argument count, types, and return type must
 * match.
 *
 * Third party code unintentionally rely on problematic fpcasts. The call
 * trampoline mitigates common occurences of bad fpcasts on Emscripten.
 */
#if defined(__EMSCRIPTEN__) && defined(PY_CALL_TRAMPOLINE)
#define _PyCFunction_TrampolineCall(meth, self, args) \
    _PyCFunctionWithKeywords_TrampolineCall( \
        (*(PyCFunctionWithKeywords)(void(*)(void))meth), self, args, NULL)
extern PyObject* _PyCFunctionWithKeywords_TrampolineCall(
    PyCFunctionWithKeywords meth, PyObject *, PyObject *, PyObject *);
#else
#define _PyCFunction_TrampolineCall(meth, self, args) \
    (meth)((self), (args))
#define _PyCFunctionWithKeywords_TrampolineCall(meth, self, args, kw) \
    (meth)((self), (args), (kw))
#endif // __EMSCRIPTEN__ && PY_CALL_TRAMPOLINE

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_OBJECT_H */
