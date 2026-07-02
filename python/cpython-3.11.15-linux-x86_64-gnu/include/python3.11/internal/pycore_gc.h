#ifndef Py_INTERNAL_GC_H
#define Py_INTERNAL_GC_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

/* GC information is stored BEFORE the object structure. */
typedef struct {
    // Pointer to next object in the list.
    // 0 means the object is not tracked
    uintptr_t _gc_next;

    // Pointer to previous object in the list.
    // Lowest two bits are used for flags documented later.
    uintptr_t _gc_prev;
} PyGC_Head;

#define _Py_AS_GC(o) ((PyGC_Head *)(o)-1)
#define _PyGC_Head_UNUSED PyGC_Head

/* True if the object is currently tracked by the GC. */
#define _PyObject_GC_IS_TRACKED(o) (_Py_AS_GC(o)->_gc_next != 0)

/* True if the object may be tracked by the GC in the future, or already is.
   This can be useful to implement some optimizations. */
#define _PyObject_GC_MAY_BE_TRACKED(obj) \
    (PyObject_IS_GC(obj) && \
        (!PyTuple_CheckExact(obj) || _PyObject_GC_IS_TRACKED(obj)))


/* Bit flags for _gc_prev */
/* Bit 0 is set when tp_finalize is called */
#define _PyGC_PREV_MASK_FINALIZED  (1)
/* Bit 1 is set when the object is in generation which is GCed currently. */
#define _PyGC_PREV_MASK_COLLECTING (2)
/* The (N-2) most significant bits contain the real address. */
#define _PyGC_PREV_SHIFT           (2)
#define _PyGC_PREV_MASK            (((uintptr_t) -1) << _PyGC_PREV_SHIFT)

// Lowest bit of _gc_next is used for flags only in GC.
// But it is always 0 for normal code.
#define _PyGCHead_NEXT(g)        ((PyGC_Head*)(g)->_gc_next)
#define _PyGCHead_SET_NEXT(g, p) _Py_RVALUE((g)->_gc_next = (uintptr_t)(p))

// Lowest two bits of _gc_prev is used for _PyGC_PREV_MASK_* flags.
#define _PyGCHead_PREV(g) ((PyGC_Head*)((g)->_gc_prev & _PyGC_PREV_MASK))
#define _PyGCHead_SET_PREV(g, p) do { \
    assert(((uintptr_t)p & ~_PyGC_PREV_MASK) == 0); \
    (g)->_gc_prev = ((g)->_gc_prev & ~_PyGC_PREV_MASK) \
        | ((uintptr_t)(p)); \
    } while (0)

#define _PyGCHead_FINALIZED(g) \
    (((g)->_gc_prev & _PyGC_PREV_MASK_FINALIZED) != 0)
#define _PyGCHead_SET_FINALIZED(g) \
    _Py_RVALUE((g)->_gc_prev |= _PyGC_PREV_MASK_FINALIZED)

#define _PyGC_FINALIZED(o) \
    _PyGCHead_FINALIZED(_Py_AS_GC(o))
#define _PyGC_SET_FINALIZED(o) \
    _PyGCHead_SET_FINALIZED(_Py_AS_GC(o))


/* GC runtime state */

/* If we change this, we need to change the default value in the
   signature of gc.collect. */
#define NUM_GENERATIONS 3
/*
   NOTE: about untracking of mutable objects.

   Certain types of container cannot participate in a reference cycle, and
   so do not need to be tracked by the garbage collector. Untracking these
   objects reduces the cost of garbage collections. However, determining
   which objects may be untracked is not free, and the costs must be
   weighed against the benefits for garbage collection.

   There are two possible strategies for when to untrack a container:

   i) When the container is created.
   ii) When the container is examined by the garbage collector.

   Tuples containing only immutable objects (integers, strings etc, and
   recursively, tuples of immutable objects) do not need to be tracked.
   The interpreter creates a large number of tuples, many of which will
   not survive until garbage collection. It is therefore not worthwhile
   to untrack eligible tuples at creation time.

   Instead, all tuples except the empty tuple are tracked when created.
   During garbage collection it is determined whether any surviving tuples
   can be untracked. A tuple can be untracked if all of its contents are
   already not tracked. Tuples are examined for untracking in all garbage
   collection cycles. It may take more than one cycle to untrack a tuple.

   Dictionaries containing only immutable objects also do not need to be
   tracked. Dictionaries are untracked when created. If a tracked item is
   inserted into a dictionary (either as a key or value), the dictionary
   becomes tracked. During a full garbage collection (all generations),
   the collector will untrack any dictionaries whose contents are not
   tracked.

   The module provides the python function is_tracked(obj), which returns
   the CURRENT tracking status of the object. Subsequent garbage
   collections may change the tracking status of the object.

   Untracking of certain containers was introduced in issue #4688, and
   the algorithm was refined in response to issue #14775.
*/

struct gc_generation {
    PyGC_Head head;
    int threshold; /* collection threshold */
    int count; /* count of allocations or collections of younger
                  generations */
};

/* Running stats per generation */
struct gc_generation_stats {
    /* total number of collections */
    Py_ssize_t collections;
    /* total number of collected objects */
    Py_ssize_t collected;
    /* total number of uncollectable objects (put into gc.garbage) */
    Py_ssize_t uncollectable;
};

struct _gc_runtime_state {
    /* List of objects that still need to be cleaned up, singly linked
     * via their gc headers' gc_prev pointers.  */
    PyObject *trash_delete_later;
    /* Current call-stack depth of tp_dealloc calls. */
    int trash_delete_nesting;

    /* Is automatic collection enabled? */
    int enabled;
    int debug;
    /* linked lists of container objects */
    struct gc_generation generations[NUM_GENERATIONS];
    PyGC_Head *generation0;
    /* a permanent generation which won't be collected */
    struct gc_generation permanent_generation;
    struct gc_generation_stats generation_stats[NUM_GENERATIONS];
    /* true if we are currently running the collector */
    int collecting;
    /* list of uncollectable objects */
    PyObject *garbage;
    /* a list of callbacks to be invoked when collection is performed */
    PyObject *callbacks;
    /* This is the number of objects that survived the last full
       collection. It approximates the number of long lived objects
       tracked by the GC.

       (by "full collection", we mean a collection of the oldest
       generation). */
    Py_ssize_t long_lived_total;
    /* This is the number of objects that survived all "non-full"
       collections, and are awaiting to undergo a full collection for
       the first time. */
    Py_ssize_t long_lived_pending;
};


extern void _PyGC_InitState(struct _gc_runtime_state *);

extern Py_ssize_t _PyGC_CollectNoFail(PyThreadState *tstate);


// Functions to clear types free lists
extern void _PyTuple_ClearFreeList(PyInterpreterState *interp);
extern void _PyFloat_ClearFreeList(PyInterpreterState *interp);
extern void _PyList_ClearFreeList(PyInterpreterState *interp);
extern void _PyDict_ClearFreeList(PyInterpreterState *interp);
extern void _PyAsyncGen_ClearFreeLists(PyInterpreterState *interp);
extern void _PyContext_ClearFreeList(PyInterpreterState *interp);

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_GC_H */
