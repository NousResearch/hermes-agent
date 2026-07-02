#ifndef Py_INTERNAL_FRAME_H
#define Py_INTERNAL_FRAME_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

/* See Objects/frame_layout.md for an explanation of the frame stack
 * including explanation of the PyFrameObject and _PyInterpreterFrame
 * structs. */


struct _frame {
    PyObject_HEAD
    PyFrameObject *f_back;      /* previous frame, or NULL */
    struct _PyInterpreterFrame *f_frame; /* points to the frame data */
    PyObject *f_trace;          /* Trace function */
    int f_lineno;               /* Current line number. Only valid if non-zero */
    char f_trace_lines;         /* Emit per-line trace events? */
    char f_trace_opcodes;       /* Emit per-opcode trace events? */
    char f_fast_as_locals;      /* Have the fast locals of this frame been converted to a dict? */
    /* The frame data, if this frame object owns the frame */
    PyObject *_f_frame_data[1];
};

extern PyFrameObject* _PyFrame_New_NoTrack(PyCodeObject *code);


/* other API */

typedef enum _framestate {
    FRAME_CREATED = -2,
    FRAME_SUSPENDED = -1,
    FRAME_EXECUTING = 0,
    FRAME_COMPLETED = 1,
    FRAME_CLEARED = 4
} PyFrameState;

enum _frameowner {
    FRAME_OWNED_BY_THREAD = 0,
    FRAME_OWNED_BY_GENERATOR = 1,
    FRAME_OWNED_BY_FRAME_OBJECT = 2
};

typedef struct _PyInterpreterFrame {
    /* "Specials" section */
    PyFunctionObject *f_func; /* Strong reference */
    PyObject *f_globals; /* Borrowed reference */
    PyObject *f_builtins; /* Borrowed reference */
    PyObject *f_locals; /* Strong reference, may be NULL */
    PyCodeObject *f_code; /* Strong reference */
    PyFrameObject *frame_obj; /* Strong reference, may be NULL */
    /* Linkage section */
    struct _PyInterpreterFrame *previous;
    // NOTE: This is not necessarily the last instruction started in the given
    // frame. Rather, it is the code unit *prior to* the *next* instruction. For
    // example, it may be an inline CACHE entry, an instruction we just jumped
    // over, or (in the case of a newly-created frame) a totally invalid value:
    _Py_CODEUNIT *prev_instr;
    int stacktop;     /* Offset of TOS from localsplus  */
    bool is_entry;  // Whether this is the "root" frame for the current _PyCFrame.
    char owner;
    /* Locals and stack */
    PyObject *localsplus[1];
} _PyInterpreterFrame;

#define _PyInterpreterFrame_LASTI(IF) \
    ((int)((IF)->prev_instr - _PyCode_CODE((IF)->f_code)))

static inline PyObject **_PyFrame_Stackbase(_PyInterpreterFrame *f) {
    return f->localsplus + f->f_code->co_nlocalsplus;
}

static inline PyObject *_PyFrame_StackPeek(_PyInterpreterFrame *f) {
    assert(f->stacktop > f->f_code->co_nlocalsplus);
    assert(f->localsplus[f->stacktop-1] != NULL);
    return f->localsplus[f->stacktop-1];
}

static inline PyObject *_PyFrame_StackPop(_PyInterpreterFrame *f) {
    assert(f->stacktop > f->f_code->co_nlocalsplus);
    f->stacktop--;
    return f->localsplus[f->stacktop];
}

static inline void _PyFrame_StackPush(_PyInterpreterFrame *f, PyObject *value) {
    f->localsplus[f->stacktop] = value;
    f->stacktop++;
}

#define FRAME_SPECIALS_SIZE ((sizeof(_PyInterpreterFrame)-1)/sizeof(PyObject *))

void _PyFrame_Copy(_PyInterpreterFrame *src, _PyInterpreterFrame *dest);

/* Consumes reference to func and locals.
   Does not initialize frame->previous, which happens
   when frame is linked into the frame stack.
 */
static inline void
_PyFrame_InitializeSpecials(
    _PyInterpreterFrame *frame, PyFunctionObject *func,
    PyObject *locals, int nlocalsplus)
{
    frame->f_func = func;
    frame->f_code = (PyCodeObject *)Py_NewRef(func->func_code);
    frame->f_builtins = func->func_builtins;
    frame->f_globals = func->func_globals;
    frame->f_locals = Py_XNewRef(locals);
    frame->stacktop = nlocalsplus;
    frame->frame_obj = NULL;
    frame->prev_instr = _PyCode_CODE(frame->f_code) - 1;
    frame->is_entry = false;
    frame->owner = FRAME_OWNED_BY_THREAD;
}

/* Gets the pointer to the locals array
 * that precedes this frame.
 */
static inline PyObject**
_PyFrame_GetLocalsArray(_PyInterpreterFrame *frame)
{
    return frame->localsplus;
}

static inline PyObject**
_PyFrame_GetStackPointer(_PyInterpreterFrame *frame)
{
    return frame->localsplus+frame->stacktop;
}

static inline void
_PyFrame_SetStackPointer(_PyInterpreterFrame *frame, PyObject **stack_pointer)
{
    frame->stacktop = (int)(stack_pointer - frame->localsplus);
}

/* Determine whether a frame is incomplete.
 * A frame is incomplete if it is part way through
 * creating cell objects or a generator or coroutine.
 *
 * Frames on the frame stack are incomplete until the
 * first RESUME instruction.
 * Frames owned by a generator are always complete.
 */
static inline bool
_PyFrame_IsIncomplete(_PyInterpreterFrame *frame)
{
    return frame->owner != FRAME_OWNED_BY_GENERATOR &&
    frame->prev_instr < _PyCode_CODE(frame->f_code) + frame->f_code->_co_firsttraceable;
}

/* For use by _PyFrame_GetFrameObject
  Do not call directly. */
PyFrameObject *
_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame *frame);

/* Gets the PyFrameObject for this frame, lazily
 * creating it if necessary.
 * Returns a borrowed referennce */
static inline PyFrameObject *
_PyFrame_GetFrameObject(_PyInterpreterFrame *frame)
{

    assert(!_PyFrame_IsIncomplete(frame));
    PyFrameObject *res =  frame->frame_obj;
    if (res != NULL) {
        return res;
    }
    return _PyFrame_MakeAndSetFrameObject(frame);
}

/* Clears all references in the frame.
 * If take is non-zero, then the _PyInterpreterFrame frame
 * may be transferred to the frame object it references
 * instead of being cleared. Either way
 * the caller no longer owns the references
 * in the frame.
 * take should  be set to 1 for heap allocated
 * frames like the ones in generators and coroutines.
 */
void
_PyFrame_Clear(_PyInterpreterFrame * frame);

int
_PyFrame_Traverse(_PyInterpreterFrame *frame, visitproc visit, void *arg);

int
_PyFrame_FastToLocalsWithError(_PyInterpreterFrame *frame);

void
_PyFrame_LocalsToFast(_PyInterpreterFrame *frame, int clear);

extern _PyInterpreterFrame *
_PyThreadState_BumpFramePointerSlow(PyThreadState *tstate, size_t size);

static inline bool
_PyThreadState_HasStackSpace(PyThreadState *tstate, size_t size)
{
    assert(
        (tstate->datastack_top == NULL && tstate->datastack_limit == NULL)
        ||
        (tstate->datastack_top != NULL && tstate->datastack_limit != NULL)
    );
    return tstate->datastack_top != NULL &&
        size < (size_t)(tstate->datastack_limit - tstate->datastack_top);
}

static inline _PyInterpreterFrame *
_PyThreadState_BumpFramePointer(PyThreadState *tstate, size_t size)
{
    if (_PyThreadState_HasStackSpace(tstate, size)) {
        _PyInterpreterFrame *res = (_PyInterpreterFrame *)tstate->datastack_top;
        tstate->datastack_top += size;
        return res;
    }
    return _PyThreadState_BumpFramePointerSlow(tstate, size);
}

void _PyThreadState_PopFrame(PyThreadState *tstate, _PyInterpreterFrame *frame);

/* Consume reference to func */
_PyInterpreterFrame *
_PyFrame_Push(PyThreadState *tstate, PyFunctionObject *func);

int _PyInterpreterFrame_GetLine(_PyInterpreterFrame *frame);

static inline
PyGenObject *_PyFrame_GetGenerator(_PyInterpreterFrame *frame)
{
    assert(frame->owner == FRAME_OWNED_BY_GENERATOR);
    size_t offset_in_gen = offsetof(PyGenObject, gi_iframe);
    return (PyGenObject *)(((char *)frame) - offset_in_gen);
}

#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_FRAME_H */
